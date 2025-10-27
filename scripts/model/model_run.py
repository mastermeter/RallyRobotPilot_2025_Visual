import sys
import torch
import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from PyQt6 import QtWidgets
from data_collector import DataCollectionUI

from model.train_tools.RobopilotCNN import RobopilotCNN
from model.train_tools.process import crop_image_ndarray, resize_image_ndarray, normalize_image_ndarray

MODEL_PATH = "scripts/model/output/robopilot_cnn_best.pth"

def preprocess_for_model(image_array):
    cropped = crop_image_ndarray(image_array, 0, 1, 0, 0.62)
    resized = resize_image_ndarray(cropped, target_size=(128, 128))
    normalized = normalize_image_ndarray(resized)

    chw = np.transpose(normalized, (2, 0, 1))  

    tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0)  

    return tensor

def outputs_to_commands(preds, threshold=0.5):
    p = preds.detach().cpu().numpy()  # [4]

    active = []
    if p[0] > threshold:
        active.append("forward")
    if p[1] > threshold:
        active.append("back")
    if p[2] > threshold:
        active.append("left")
    if p[3] > threshold:
        active.append("right")

    return active


class NNMsgProcessor:
    def __init__(self, model_path="scripts/model/output/robopilot_cnn_best.pth", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model = RobopilotCNN(output_size=4).to(self.device)
        self.model.eval() 

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print("[NNMsgProcessor] Model weights loaded")

        self.last_state = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False,
        }

    def _infer_controls(self, frame_rgb):
        if frame_rgb is None:
            return []

        input_tensor = preprocess_for_model(frame_rgb).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)
        preds = preds[0]

        active_cmds = outputs_to_commands(preds)
        print(active_cmds)

        return active_cmds

    def _send_command_state(self, data_collector, command, should_be_on):
        prev = self.last_state[command]
        if prev != should_be_on:
            self.last_state[command] = should_be_on
            data_collector.onCarControlled(command, should_be_on)

    def process_message(self, message, data_collector):
        frame_rgb = message.image
        active_now = self._infer_controls(frame_rgb)

        for cmd in ["forward", "back", "left", "right"]:
            self._send_command_state(
                data_collector,
                cmd,
                cmd in active_now
            )

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = NNMsgProcessor(
        model_path=MODEL_PATH,
        device=None 
    )

    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
