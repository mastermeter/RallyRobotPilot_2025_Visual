import sys, os, time
import torch
import numpy as np
from collections import deque
from PyQt6 import QtWidgets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from data_collector import DataCollectionUI
from model.train_tools.RobopilotCNNLSTM import RobopilotCNNLSTM
from model.train_tools.process import (
    normalize_image_ndarray,
    resize_image_ndarray,
    crop_image_ndarray,
)

MODEL_PATH = "scripts/model/output/robopilot_cnn_best_90k_11SEQ.pth"
SEQ_LEN = 11


def preprocess_for_model(image_array):
    resized   = resize_image_ndarray(image_array, target_size=(128, 128))
    normalized = normalize_image_ndarray(resized)
    return normalized  # (H, W, 3), float32


class NNMsgProcessor:
    TH_LR_ON, TH_LR_OFF = 0.30, 0.30 #0.50
    TH_F_ON, TH_F_OFF = 0.65, 0.35

    COAST_TH = 0.80
    COAST_MS = 140

    STARTUP_MS = 800
    STARTUP_MIN_PF = TH_F_ON

    def __init__(self, model_path=MODEL_PATH, device=None, debug=False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.debug = debug
        print(f"[RUN] Using device: {self.device}")

        self.model = RobopilotCNNLSTM(
            in_channels=3,
            seq_len=SEQ_LEN,
            cnn_feature_dim=256,
            lstm_hidden_dim=128,
            lstm_num_layers=1,
            output_size=3,
            dropout_rate=0.3,
        ).to(self.device)
        self.model.eval()

        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print("[RUN] CNN+LSTM model weights loaded.")

        self.state = {"forward": False, "back": False, "left": False, "right": False}

        self.hold_left = False
        self.hold_right = False
        self.hold_forward = False

        self.coast_until = 0.0
        self.t0_ms = None

        self.buffer = deque(maxlen=SEQ_LEN)

    def _send(self, ui, key, want):
        cur = self.state[key]
        if cur != want:
            self.state[key] = want
            ui.onCarControlled(key, want)

    def _hyst_lr(self, p_left, p_right):
        print(f"p_l={p_left:.2f} p_r={p_right:.2f}")

        if self.hold_left:
            self.hold_left = p_left >= self.TH_LR_OFF
        else:
            self.hold_left = (p_left >= self.TH_LR_ON) and (p_left >= p_right)

        if self.hold_right:
            self.hold_right = p_right >= self.TH_LR_OFF
        else:
            self.hold_right = (p_right >= self.TH_LR_ON) and (p_right >= p_left)

        if self.hold_left and self.hold_right:
            if p_left > p_right:
                self.hold_right = False
            else:
                self.hold_left = False

        return self.hold_left, self.hold_right

    def _hyst_forward(self, p_f):
        if self.hold_forward:
            self.hold_forward = (p_f >= self.TH_F_OFF)
        else:
            self.hold_forward = (p_f >= self.TH_F_ON)
        return self.hold_forward

    def _infer_probs(self):
        if len(self.buffer) == 0:
            return None

        if len(self.buffer) < SEQ_LEN:
            first = self.buffer[0]
            pad_needed = SEQ_LEN - len(self.buffer)
            frames = [first] * pad_needed + list(self.buffer)
        else:
            frames = list(self.buffer)[-SEQ_LEN:]

        seq = np.stack(frames, axis=0)  # (T, H, W, 3)
        seq = np.transpose(seq, (0, 3, 1, 2))  # (T, 3, H, W)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        with torch.no_grad():
            logits = self.model(x)[0]                  # (3,)
            probs = torch.sigmoid(logits).cpu().numpy() # [0,1]

        return probs.tolist()  # [p_f, p_l, p_r]

    def process_message(self, message, ui: DataCollectionUI):
        if self.t0_ms is None:
            self.t0_ms = time.time() * 1000.0

        if message.image is None:
            return
        img_norm = preprocess_for_model(message.image)   # (H, W, 3), float32
        self.buffer.append(img_norm)

        probs = self._infer_probs()
        if probs is None:
            return

        p_f, p_l, p_r = probs

        hold_l, hold_r = self._hyst_lr(p_l, p_r)
        turn_intensity = max(p_l, p_r)

        want_f = self._hyst_forward(p_f)

        now_ms = time.time() * 1000.0
        if (now_ms - self.t0_ms) < self.STARTUP_MS and not want_f and p_f < self.STARTUP_MIN_PF:
            want_f = True
            self.hold_forward = True

        if turn_intensity > self.COAST_TH and now_ms >= self.coast_until:
            self.coast_until = now_ms + self.COAST_MS
            want_f = False
            self.hold_forward = False
        if now_ms < self.coast_until:
            want_f = False

        self._send(ui, "left",  hold_l)
        self._send(ui, "right", hold_r)
        self._send(ui, "forward", want_f)
        self._send(ui, "back", False)

        if self.debug:
            print(
                f"P(f,l,r)=({p_f:.2f},{p_l:.2f},{p_r:.2f})  "
                f"LR=({hold_l},{hold_r})  F={want_f}"
            )


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    sys.excepthook = except_hook
    app = QtWidgets.QApplication(sys.argv)
    brain = NNMsgProcessor(MODEL_PATH, debug=False)
    win = DataCollectionUI(brain.process_message)
    win.show()
    app.exec()
