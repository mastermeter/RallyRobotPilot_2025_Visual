import sys, time, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from PyQt6 import QtWidgets
from data_collector import DataCollectionUI

class ForwardOnlyBrain:
    """
    Avance tout droit avec le même régulateur que le fichier run :
    - Kick de départ si la voiture est quasi immobile
    - Bande morte autour d'une vitesse cible (V_SET ± BAND)
    - Anti-spam minimal sur les changements d'état
    """
    # ==== mêmes constantes que dans ton run ====
    V_SET = 7.5          # vitesse cible (m/s)
    BAND  = 1.0          # bande morte (m/s)
    KICK_MS = 350        # durée du kick d'accélération initial
    KICK_SPEED_TH = 0.2  # seuil "quasi immobile"

    def __init__(self, min_hold_s: float = 0.06, debug: bool = True):
        self.state = {"forward": False, "back": False, "left": False, "right": False}

        # régulation forward
        self.forward_hold = False
        self.first_tick_ms = None
        self.kick_until_ms = 0.0

        # anti-spam
        self.MIN_HOLD_S = float(min_hold_s)
        self.last_change_t = 0.0

        self.debug = debug

    def _send(self, ui, key, want):
        """n'envoie qu'en cas de changement + anti-spam minimal"""
        now = time.time()
        if self.state[key] == want:
            return
        if now - self.last_change_t < self.MIN_HOLD_S:
            return
        self.state[key] = want
        ui.onCarControlled(key, want)
        self.last_change_t = now
        if self.debug:
            print(f"[AUTO] {key} -> {want}")

    def _update_forward_from_speed(self, speed):
        """copie exacte de la logique du run pour déterminer forward_hold"""
        now_ms = time.time() * 1000.0
        if self.first_tick_ms is None:
            self.first_tick_ms = now_ms

        # Kick de départ si on est quasi immobile dans les 2 premières secondes
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
        # sinon on conserve l'état actuel (bande morte)

    def process_message(self, message, data_collector):
        # lire la vitesse du message (comme dans run)
        spd = float(getattr(message, "car_speed", 0.0))
        self._update_forward_from_speed(spd)

        # n'envoyer que les commandes nécessaires
        self._send(data_collector, "forward", self.forward_hold)
        self._send(data_collector, "back", False)

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

    # Optionnel : on repart propre
    ni = ui.network_interface
    _send_cmd(ni, "release all")
    _send_cmd(ni, "reset")

    app.exec()
