# checkpoints.py
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ursina import Entity, Vec3, color, destroy, Text

# ==== Utilitaires géométrie ====


def v2(x, z):  # on travaille dans le plan XZ d'Ursina
    return np.array([x, z], dtype=float)


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def seg_intersect(a, b, c, d):
    """Retourne True si les segments AB et CD s'intersectent (2D)."""

    def orient(p, q, r):
        return np.cross(q - p, r - p)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # Cas colinéaire : check par projections approximatives
    if (o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0):
        def proj_range(p, q):
            return (min(p[0], q[0]), max(p[0], q[0]),
                    min(p[1], q[1]), max(p[1], q[1]))

        ax1, ax2, ay1, ay2 = proj_range(a, b)
        cx1, cx2, cy1, cy2 = proj_range(c, d)
        return not (ax2 < cx1 or cx2 < ax1 or ay2 < cy1 or cy2 < ay1)

    return (o1 * o2 <= 0) and (o3 * o4 <= 0)


# ==== Données de checkpoint ====


@dataclass
class Checkpoint:
    # centre dans XZ, normale = direction d'avancement (2D), demi-longueur (zone utile pour l'intersection)
    x: float
    z: float
    nx: float
    nz: float
    half_len: float = 6.0  # à ajuster selon la largeur de piste

    def center_v(self) -> np.ndarray:
        return v2(self.x, self.z)

    def normal_v(self) -> np.ndarray:
        return norm(v2(self.nx, self.nz))

    def endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Endpoints du petit segment de checkpoint utilisé pour
        la détection géométrique (on peut garder ça assez court).
        """
        n = self.normal_v()
        t_perp = np.array([-n[1], n[0]])  # rotation +90°
        p = self.center_v()
        return (p - self.half_len * t_perp, p + self.half_len * t_perp)


# ==== Manager Ursina ====


class CheckpointManager(Entity):
    """
    Système de checkpoints 100% manuel.

    - Touche 'n' : pose un checkpoint à la position/orientation actuelle de la voiture.
    - Touche 'u' : undo (supprime le dernier checkpoint).
    - Touche 'p' : sauvegarde dans tracks/<track_name>/checkpoints.json.
    - Touche 'l' : recharge depuis ce fichier.
    - Touche 'v' : montre/cache les lignes.

    La détection de franchissement se fait en continu quand des checkpoints existent.
    """

    def __init__(
        self,
        car,
        track_name: str,
        save_dir: str = "tracks",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.car = car
        self.track_name = track_name
        self.save_path = Path(save_dir) / track_name / "checkpoints.json"
        self.visible_lines = True

        # checkpoints
        self.checkpoints: List[Checkpoint] = []
        self._line_entities: List[Entity] = []


        # état course
        self.next_idx = 0
        self.lap_count = 0
        self.last_cross_t = 0.0
        self.cooldown = 0.25  # s, anti-double comptage
        self.prev_pos = self._car_pos2d()
        self.has_started = False

        # lignes visuelles très longues
        self.long_line_length = 10.0

        # UI mini
        self.status_text = Text(text="", scale=0.8, position=(-0.875, .475))

        # tenter de charger existants
        self.load()

        # afficher
        self._refresh_lines()

    def on_checkpoint_passed(self, idx: int, cp: Checkpoint):
        print(f"[Checkpoints] Passed CP {idx} at ({cp.x:.2f}, {cp.z:.2f})")

    def on_lap_completed(self, lap: int):
        print(f"[Checkpoints] Lap {lap} completed!")

    # --- helpers car ---

    def _car_pos2d(self) -> np.ndarray:
        p = self.car.world_position  # Vec3
        return v2(p.x, p.z)

    def _car_forward2d(self) -> np.ndarray:
        # direction avant de la voiture en XZ (on projette l'orientation)
        f: Vec3 = self.car.forward
        return norm(v2(f.x, f.z))

    # --- persistence ---

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(c) for c in self.checkpoints]
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Checkpoints] saved {len(self.checkpoints)} → {self.save_path.resolve()}")

    def load(self):
        if self.save_path.exists():
            with open(self.save_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.checkpoints = [Checkpoint(**d) for d in raw]
            print(f"[Checkpoints] loaded {len(self.checkpoints)} from {self.save_path.resolve()}")
        else:
            print(f"[Checkpoints] no file at {self.save_path.resolve()} (starting empty).")

    # --- visuel ---

    def _clear_lines(self):
        for e in self._line_entities:
            destroy(e)
        self._line_entities.clear()

    def _refresh_lines(self):
        self._clear_lines()
        if not self.visible_lines:
            return

        for i, cp in enumerate(self.checkpoints):
            n = cp.normal_v()

            # yaw de la normale (direction dans laquelle il faut passer le checkpoint)
            yaw_deg = math.degrees(math.atan2(n[0], n[1]))

            # 1) Trait principal (largeur du checkpoint)
            L = cp.half_len * 2.0   # longueur visible = largeur utile

            line_ent = Entity(
                model="cube",
                position=Vec3(cp.x, 0.02, cp.z),
                scale=(L, 0.08, 0.08),
                rotation_y=yaw_deg,          # même orientation que la normale
                color=color.azure if i else color.orange,
            )
            self._line_entities.append(line_ent)

    # --- gestion manuelle des checkpoints ---

    def add_checkpoint_at_car(self):
        fwd2d = self._car_forward2d()

        # Valeurs par défaut si on n'arrive pas à lire les rayons
        center_pos = self._car_pos2d()
        half_len = 6.0

        # Essayer d'utiliser les rayons pour récupérer largeur + centre de piste
        mrs = getattr(self.car, "multiray_sensor", None)
        if mrs is not None and hasattr(mrs, "get_edge_points"):
            width, center = mrs.get_edge_points()
            if width is not None and center is not None and width > 0:
                center_pos = center
                half_len = (width * 0.5) * 0.95  # un peu moins large que la piste
                print(f"[Checkpoints] width≈{width:.2f}, center=({center_pos[0]:.2f},{center_pos[1]:.2f}), half_len={half_len:.2f}")
            else:
                print("[Checkpoints] no valid width/center from rays, using defaults.")
        else:
            print("[Checkpoints] multiray_sensor or get_edge_points() not available, using defaults.")

        cp = Checkpoint(
            x=float(center_pos[0]),
            z=float(center_pos[1]),
            nx=float(fwd2d[0]),
            nz=float(fwd2d[1]),
            half_len=half_len,
        )

        self.checkpoints.append(cp)
        self._refresh_lines()
        print(f"[Checkpoints] added at ({cp.x:.2f}, {cp.z:.2f})")


    # --- course ---

    def reset_race(self):
        self.next_idx = 0
        self.lap_count = 0
        self.last_cross_t = 0.0
        self.prev_pos = self._car_pos2d()
        self.has_started = False
        print("[Checkpoints] Race state reset.")

    def toggle_visible(self):
        self.visible_lines = not self.visible_lines
        self._refresh_lines()

    def _update_status_text(self):
        txt = f"[RACE] CP:{self.next_idx}/{len(self.checkpoints)}  Laps:{self.lap_count}"
        self.status_text.text = txt

    # --- boucle de mise à jour Ursina ---

    def update(self):
        pos2d = self._car_pos2d()

        # course: détection de franchissement
        if self.checkpoints:
            i = self.next_idx
            cp = self.checkpoints[i]

            n = cp.normal_v()
            p = cp.center_v()

            # distances signées avant/après
            d_prev = np.dot(n, self.prev_pos - p)
            d_now  = np.dot(n, pos2d      - p)

            # segment du checkpoint
            a, b = cp.endpoints()
            crossed_geom = seg_intersect(self.prev_pos, pos2d, a, b)

            # franchissement dans le bon sens + intersection + cooldown
            if d_prev < 0 and d_now > 0 and crossed_geom:
                tnow = time.time()
                if (tnow - self.last_cross_t) > self.cooldown:
                    self.last_cross_t = tnow

                    # Index du CP qu'on vient de franchir
                    passed_idx = i

                    # Hook: checkpoint franchi
                    self.on_checkpoint_passed(passed_idx, cp)

                    # Avancer au prochain CP
                    self.next_idx = (self.next_idx + 1) % len(self.checkpoints)

                    # Gestion des tours : basé sur CP0
                    if passed_idx == 0:
                        if not self.has_started:
                            # Premier passage de CP0 : on démarre la course
                            self.has_started = True
                            print("[Checkpoints] Race started at CP 0")
                        else:
                            # Passages suivants de CP0 : tours complets
                            self.lap_count += 1
                            self.on_lap_completed(self.lap_count)


        # maj position précédente
        self.prev_pos = pos2d
        self._update_status_text()


    # --- gestion des touches (one-shot) ---

    def input(self, key):
        """
        Géré par Ursina : appelé une fois par pression de touche.
        """
        # Ajouter un checkpoint
        if key == 'n':
            self.add_checkpoint_at_car()

        # Supprimer le dernier checkpoint
        elif key == 'u':
            if self.checkpoints:
                removed = self.checkpoints.pop()
                self._refresh_lines()
                print(f"[Checkpoints] removed at ({removed.x:.2f}, {removed.z:.2f})")
            else:
                print("[Checkpoints] no checkpoints to remove.")

        # Sauvegarder
        elif key == 'p':
            self.save()

        # Charger
        elif key == 'l':
            self.load()
            self._refresh_lines()

        # Visibilité
        elif key == 'v':
            self.toggle_visible()
