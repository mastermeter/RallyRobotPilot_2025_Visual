import sys, os, time
import torch
import numpy as np
from PyQt6 import QtWidgets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from data_collector import DataCollectionUI
from model.train_tools.RobopilotCNN import RobopilotCNN
from model.train_tools.process import normalize_image_ndarray, resize_image_ndarray, crop_image_ndarray

MODEL_PATH = "scripts/model/output/robopilot_cnn_best.pth"

def preprocess_for_model(image_array):
    # même préproc que l’entraînement
    #cropped   = crop_image_ndarray(image_array, 0, 1, 0, 0.62)
    resized   = resize_image_ndarray(image_array, target_size=(128, 128))
    normalized= normalize_image_ndarray(resized)
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.tensor(chw, dtype=torch.float32).unsqueeze(0)

class NNMsgProcessor:
    TH_ON, TH_OFF = 0.3, 0.2

    V_SET = 7.5        # speed target (m/s)
    BAND  = 1.0       # speed deadband (m/s)

    KICK_MS = 350 
    KICK_SPEED_TH = 0.2

    def __init__(self, model_path=MODEL_PATH, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model = RobopilotCNN(output_size=2).to(self.device)
        self.model.eval()
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.state = {"forward": False, "back": False, "left": False, "right": False}
        self.hold_left = False
        self.hold_right = False

        self.forward_hold = False
        self.first_tick_ms = None
        self.kick_until_ms = 0.0

        print("[NNMsgProcessor] Loaded (steer L/R + speed governor).")

    def _send(self, data_collector, key, want):
        cur = self.state[key]
        if cur != want:
            self.state[key] = want
            data_collector.onCarControlled(key, want)

    def _hysteresis_steer(self, p_left, p_right):
        print(f"Steer probabilities: L={p_left:.3f} R={p_right:.3f}")
        # gauche
        if self.hold_left:
            self.hold_left = (p_left >= self.TH_OFF) and (p_right < self.TH_ON or p_left >= p_right)
        else:
            self.hold_left = (p_left >= self.TH_ON) and (p_left >= p_right)
        if self.hold_right:
            self.hold_right = (p_right >= self.TH_OFF) and (p_left < self.TH_ON or p_right >= p_left)
        else:
            self.hold_right = (p_right >= self.TH_ON) and (p_right >= p_left)
        if self.hold_left and self.hold_right:
            if p_left > p_right: self.hold_right = False
            else:                self.hold_left  = False
        return self.hold_left, self.hold_right

    def _infer_steer(self, frame_rgb):
        if frame_rgb is None:
            return False, False
        x = preprocess_for_model(frame_rgb).to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]                  
            p_left, p_right = torch.sigmoid(logits).cpu().numpy().tolist()
        return self._hysteresis_steer(p_left, p_right)

    def _update_forward_from_speed(self, speed):
        now_ms = time.time() * 1000.0

        if self.first_tick_ms is None:
            self.first_tick_ms = now_ms
        if speed < self.KICK_SPEED_TH and (now_ms - self.first_tick_ms) < 2000:
            self.kick_until_ms = max(self.kick_until_ms, now_ms + self.KICK_MS)

        if now_ms < self.kick_until_ms:
            self.forward_hold = True
            return

        lo = self.V_SET - self.BAND
        hi = self.V_SET + self.BAND
        if speed < lo:
            self.forward_hold = True
        elif speed > hi:
            self.forward_hold = False

    def process_message(self, message, data_collector):
        hold_l, hold_r = self._infer_steer(message.image)
        self._send(data_collector, "left",  hold_l)
        self._send(data_collector, "right", hold_r)

        spd = float(getattr(message, "car_speed", 0.0))
        self._update_forward_from_speed(spd)
        self._send(data_collector, "forward", self.forward_hold)

        self._send(data_collector, "back", False)

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    sys.excepthook = except_hook
    app = QtWidgets.QApplication(sys.argv)
    brain = NNMsgProcessor(MODEL_PATH)
    win = DataCollectionUI(brain.process_message)
    win.show()
    app.exec()
