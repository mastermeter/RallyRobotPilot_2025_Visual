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
from model.train_tools.old_LR_only.RobopilotCNN import RobopilotCNN
from model.train_tools.process import (
    normalize_image_ndarray,
    resize_image_ndarray,
    crop_image_ndarray,
)

MODEL_LSTM_PATH = "scripts/model/output/robopilot_cnn_best_90k_11SEQ.pth"
MODEL_STEER_PATH = "scripts/model/output/robopilot_cnn_LR_GOAT.pth"
SEQ_LEN = 11


def preprocess_frame_for_buffer(image_array):
    resized = resize_image_ndarray(image_array, target_size=(128, 128))
    normalized = normalize_image_ndarray(resized)
    return normalized


def preprocess_frame_for_steer(image_array_norm):
    chw = np.transpose(image_array_norm, (2, 0, 1))
    return torch.tensor(chw, dtype=torch.float32).unsqueeze(0)


class HybridNNProcessor:
    TH_F_ON, TH_F_OFF = 0.65, 0.35
    COAST_TH = 0.80
    COAST_MS = 140
    STARTUP_MS = 800
    STARTUP_MIN_PF = TH_F_ON

    TH_LR_ON, TH_LR_OFF = 0.3, 0.2
    
    def __init__(self, model_lstm_path, model_steer_path, device=None, debug=False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.debug = debug
        print(f"[RUN] Using device: {self.device}")

        self.model_forward = RobopilotCNNLSTM(
            in_channels=3, seq_len=SEQ_LEN, cnn_feature_dim=256,
            lstm_hidden_dim=128, lstm_num_layers=1, output_size=3, dropout_rate=0.3,
        ).to(self.device)
        self.model_forward.eval()
        ckpt_fwd = torch.load(model_lstm_path, map_location=self.device)
        self.model_forward.load_state_dict(ckpt_fwd["model_state_dict"])
        print("[RUN] CNN+LSTM (Forward) model weights loaded.")

        self.model_steer = RobopilotCNN(output_size=2).to(self.device)
        self.model_steer.eval()
        ckpt_steer = torch.load(model_steer_path, map_location=self.device)
        self.model_steer.load_state_dict(ckpt_steer["model_state_dict"])
        print("[RUN] CNN (Steer) model weights loaded.")

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

    def _hyst_steer(self, p_left, p_right):
        if self.hold_left:
            self.hold_left = (p_left >= self.TH_LR_OFF) and (p_right < self.TH_LR_ON or p_left >= p_right)
        else:
            self.hold_left = (p_left >= self.TH_LR_ON) and (p_left >= p_right)
        if self.hold_right:
            self.hold_right = (p_right >= self.TH_LR_OFF) and (p_left < self.TH_LR_ON or p_right >= p_left)
        else:
            self.hold_right = (p_right >= self.TH_LR_ON) and (p_right >= p_left)
        if self.hold_left and self.hold_right:
            if p_left > p_right: 
                self.hold_right = False
            else:                
                self.hold_left  = False
        return self.hold_left, self.hold_right

    def _hyst_forward(self, p_f):
        if self.hold_forward:
            self.hold_forward = (p_f >= self.TH_F_OFF)
        else:
            self.hold_forward = (p_f >= self.TH_F_ON)
        return self.hold_forward

    def _infer_forward_prob(self):
        if len(self.buffer) == 0:
            return None

        if len(self.buffer) < SEQ_LEN:
            first = self.buffer[0]
            pad_needed = SEQ_LEN - len(self.buffer)
            frames = [first] * pad_needed + list(self.buffer)
        else:
            frames = list(self.buffer)[-SEQ_LEN:]

        seq = np.stack(frames, axis=0)
        seq = np.transpose(seq, (0, 3, 1, 2))
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model_forward(x)[0]
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs.tolist()[0] 

    def _infer_steer_probs(self, frame_h_w_c):
        x = preprocess_frame_for_steer(frame_h_w_c).to(self.device)
        with torch.no_grad():
            logits = self.model_steer(x)[0]
            p_left, p_right = torch.sigmoid(logits).cpu().numpy().tolist()
        return p_left, p_right

    def process_message(self, message, ui: DataCollectionUI):
        if self.t0_ms is None:
            self.t0_ms = time.time() * 1000.0

        if message.image is None:
            return
            
        img_norm = preprocess_frame_for_buffer(message.image)
        self.buffer.append(img_norm) 

        p_f = self._infer_forward_prob()
        p_l, p_r = self._infer_steer_probs(img_norm)

        if p_f is None:
            return

        hold_l, hold_r = self._hyst_steer(p_l, p_r)
        want_f = self._hyst_forward(p_f)

        now_ms = time.time() * 1000.0
        turn_intensity = max(p_l, p_r) 

        if (now_ms - self.t0_ms) < self.STARTUP_MS and not want_f and p_f < self.STARTUP_MIN_PF:
            want_f = True
            self.hold_forward = True

        if turn_intensity > self.COAST_TH and now_ms >= self.coast_until:
            self.coast_until = now_ms + self.COAST_MS
            want_f = False
            self.hold_forward = False
        if now_ms < self.coast_until:
            want_f = False

        self._send(ui, "left",   hold_l)
        self._send(ui, "right",  hold_r)
        self._send(ui, "forward", want_f)
        self._send(ui, "back",   False)

        if self.debug:
            print(
                f"P(f)={p_f:.2f} P(l,r)=({p_l:.2f},{p_r:.2f})  "
                f"LR=({hold_l},{hold_r})  F={want_f}"
            )


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    sys.excepthook = except_hook
    app = QtWidgets.QApplication(sys.argv)
    
    brain = HybridNNProcessor(
        MODEL_LSTM_PATH, 
        MODEL_STEER_PATH, 
        debug=False
    )
    
    win = DataCollectionUI(brain.process_message)
    win.show()
    app.exec()