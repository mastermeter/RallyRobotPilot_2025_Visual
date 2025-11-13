# scripts/autopilot_forward_only.py
import sys, time
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from PyQt6 import QtWidgets
from data_collector import DataCollectionUI

class ForwardOnlyBrain:
    """Maintient 'forward' enfoncé et relâche tout le reste."""
    def __init__(self, min_hold_s: float = 0.06, debug: bool = True):
        self.state = {"forward": False, "back": False, "left": False, "right": False}
        self.MIN_HOLD_S = float(min_hold_s)
        self.last_change_t = 0.0
        self.debug = debug

    def process_message(self, message, data_collector):
        desired = {"forward": True, "back": False, "left": False, "right": False}
        now = time.time()
        if now - self.last_change_t < self.MIN_HOLD_S:
            return

        changed = False
        for key in ["forward", "back", "left", "right"]:
            cur, want = self.state[key], desired[key]
            if cur != want:
                data_collector.onCarControlled(key, want)
                self.state[key] = want
                changed = True

        if changed:
            self.last_change_t = now
            if self.debug:
                print(f"[AUTO] sent: {self.state}")

def _send_cmd(ni, cmd: str):
    cmd = cmd.strip()
    if not cmd.endswith(";"):
        cmd += ";"
    ni.send_cmd(cmd)

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    sys.excepthook = except_hook
    app = QtWidgets.QApplication(sys.argv)

    brain = ForwardOnlyBrain(min_hold_s=0.06, debug=True)
    ui = DataCollectionUI(brain.process_message)
    ui.show()

    # (Optionnel) Nettoyage et reset au démarrage
    ni = ui.network_interface
    _send_cmd(ni, "release all")
    _send_cmd(ni, "reset")

    app.exec()
