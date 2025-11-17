import os.path

from rallyrobopilot import *

from PyQt6.QtCore import Qt, QTimer
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6 import uic
from model.train_tools.process import crop_image_ndarray, resize_image_ndarray

import pickle
import lzma

import numpy as np
import cv2

def pack_snapshots_to_arrays(snaps, crop_ratio=0.62, out_size=(128,128), keep_images=True):
    """
    Convertit une liste de SensingSnapshot en arrays NumPy prêts à être sauvegardés.
    - Images: crop (haut jusqu'à crop_ratio) + resize via helpers, en uint8 (pas de normalisation ici).
    - Le reste (controls, speed, angle, position, rays) en formats compacts.
    """
    imgs, speeds, angles, poss, rays_list, ctrls = [], [], [], [], [], []

    # Longueur des rayons (R)
    R = 0
    for s in snaps:
        if s.raycast_distances is not None and len(s.raycast_distances) > 0:
            R = len(s.raycast_distances)
            break

    for s in snaps:
        # Controls -> uint8
        ctrls.append([
            int(bool(s.current_controls[0])),
            int(bool(s.current_controls[1])),
            int(bool(s.current_controls[2])),
            int(bool(s.current_controls[3])),
        ])

        # Scalars / vecteurs
        speeds.append(np.float32(s.car_speed))
        angles.append(np.float32(s.car_angle))
        poss.append([
            np.float32(s.car_position[0]),
            np.float32(s.car_position[1]),
            np.float32(s.car_position[2]),
        ])

        # Rays -> float32 taille fixe R (pad si nécessaire)
        r = np.asarray(s.raycast_distances, dtype=np.float32) if s.raycast_distances is not None else np.zeros((0,), np.float32)
        if r.size != R:
            rr = np.zeros((R,), np.float32)
            rr[:min(R, r.size)] = r[:min(R, r.size)]
            r = rr
        rays_list.append(r)

        # Image -> crop + resize (uint8) si demandé
        if keep_images and (s.image is not None):
            #img = crop_image_ndarray(s.image, 0, 1, 0, crop_ratio)          # garde le haut
            img = resize_image_ndarray(s.image, target_size=out_size)           # (W,H)
            imgs.append(img.astype(np.uint8))

    out = {
        "controls":  np.asarray(ctrls,  dtype=np.uint8),     # (N,4)
        "speed":     np.asarray(speeds, dtype=np.float32),   # (N,)
        "angle":     np.asarray(angles, dtype=np.float32),   # (N,)
        "position":  np.asarray(poss,   dtype=np.float32),   # (N,3)
        "rays":      np.asarray(rays_list, dtype=np.float32),# (N,R)
        "ray_count": np.int32(R),
        "crop_ratio": np.float32(crop_ratio),
        "out_h": np.int32(out_size[1]),  # out_size = (W,H)
        "out_w": np.int32(out_size[0]),
    }
    out["images"] = np.asarray(imgs, dtype=np.uint8) if (keep_images and len(imgs) > 0) else np.zeros((0,1,1,3), dtype=np.uint8)
    return out

class DataCollectionUI(QtWidgets.QMainWindow):
    def __init__(self, message_processing_callback = None):
        super().__init__()

        uic.loadUi("scripts/DataCollector.ui", self)

        buttons = [self.forwardButton, self.backwardButton, self.rightButton, self.leftButton]
        self.command_directions = { "w":"forward", "s":"back", "d":"right", "a":"left" }

        self.forwardButton.pressed.connect(lambda : self.onCarControlled("forward", True))
        self.forwardButton.released.connect(lambda : self.onCarControlled("forward", False))

        self.backwardButton.pressed.connect(lambda : self.onCarControlled("back", True))
        self.backwardButton.released.connect(lambda : self.onCarControlled("back", False))

        self.rightButton.pressed.connect(lambda : self.onCarControlled("right", True))
        self.rightButton.released.connect(lambda : self.onCarControlled("right", False))

        self.leftButton.pressed.connect(lambda : self.onCarControlled("left", True))
        self.leftButton.released.connect(lambda : self.onCarControlled("left", False))

        self.recordDataButton.clicked.connect(self.toggleRecord)
        self.resetButton.clicked.connect(self.resetNForget)

        self.autopiloting = False
        def toggle_autopilot():
            self.autopiloting = not self.autopiloting
            self.AutopilotButton.setText("AutoPilot:\n" + ("ON" if self.autopiloting else "OFF"))

        self.AutopilotButton.clicked.connect(toggle_autopilot)
        self.message_processing_callback = message_processing_callback

        self.saveRecordButton.clicked.connect(self.saveRecord)

        self.network_interface = NetworkDataCmdInterface(self.collectMsg)

        self.timer = QTimer()
        self.timer.timeout.connect(self.network_interface.recv_msg)
        self.timer.start(25)

        self.saving_worker = None

        self.recording = False

        self.recorded_data = []
    def collectMsg(self, msg):
        if self.recording:
            if not self.saveImgCheckBox.isChecked():
                msg.image = None

            self.recorded_data.append(msg)
            self.nbrSnapshotSaved.setText(str(len(self.recorded_data)))

        if self.autopiloting:
            if self.message_processing_callback is not None:
                self.message_processing_callback(msg, self)

    def resetNForget(self):

        if len(self.recorded_data) == 0:
            return

        nbr_snapshots_to_forget = self.forgetSnapshotNumber.value() if len(self.recorded_data) > self.forgetSnapshotNumber.value() else len(self.recorded_data)-1

        self.recorded_data = self.recorded_data[:-nbr_snapshots_to_forget]
        self.nbrSnapshotSaved.setText(str(len(self.recorded_data)))

        self.network_interface.send_cmd("set position "+ str(self.recorded_data[-1].car_position)[1:-1].replace(" ","")+";")
        self.network_interface.send_cmd("set rotation "+ str(self.recorded_data[-1].car_angle)+";")
        self.network_interface.send_cmd("reset;")

        self.toggleRecord()

    def toggleRecord(self):
        self.recording = not self.recording
        self.recordDataButton.setText("Recording..." if self.recording else "Record")
    def onCarControlled(self, direction, start):
        command_types = ["release", "push"]
        self.network_interface.send_cmd(command_types[start] + " " + direction+";")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        if isinstance(event, QtGui.QKeyEvent):
            key_text = event.text()
            if key_text in self.command_directions:
                self.onCarControlled(self.command_directions[key_text], True)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        if isinstance(event, QtGui.QKeyEvent):
            key_text = event.text()
            if key_text in self.command_directions:
                self.onCarControlled(self.command_directions[key_text], False)

    def saveRecord(self):
        if self.saving_worker is not None:
            print("[X] Already saving !")
            return

        if len(self.recorded_data) == 0:
            print("[X] No data to save !")
            return

        if self.recording:
            self.toggleRecord()

        self.saveRecordButton.setText("Saving ...")

        record_name = "record_%d.npz"
        fid = 0
        while os.path.exists(record_name % fid):
            fid += 1

        class ThreadedSaver(QtCore.QThread):
            def __init__(self, path, data, crop_ratio=0.62, out_size=(128,128), keep_images=True):
                super().__init__()
                self.path = path
                self.data = data
                self.crop_ratio = crop_ratio
                self.out_size = out_size
                self.keep_images = keep_images

            def run(self):
                arrays = pack_snapshots_to_arrays(
                    self.data,
                    crop_ratio=self.crop_ratio,
                    out_size=self.out_size,
                    keep_images=self.keep_images
                )
                # Écriture TRÈS rapide : Zip Deflate interne
                np.savez_compressed(self.path, **arrays)

        self.saving_worker = ThreadedSaver(record_name % fid, self.recorded_data,
                                   crop_ratio=0.62, out_size=(128,128),
                                   keep_images=self.saveImgCheckBox.isChecked())
        self.recorded_data = []
        self.nbrSnapshotSaved.setText("0")
        self.saving_worker.finished.connect(self.onRecordSaveDone)
        self.saving_worker.start()

    def onRecordSaveDone(self):
        print("[+] Recorded data saved to", self.saving_worker.path)
        self.saving_worker = None
        self.saveRecordButton.setText("Save")

if __name__ == "__main__":

    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    data_window = DataCollectionUI()
    data_window.show()

    app.exec()
