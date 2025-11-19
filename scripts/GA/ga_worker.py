# scripts/ga_worker.py
import sys, os
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

""" from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type none')        
loadPrcFileData('', 'audio-library-name null') """

from ursina import Entity, held_keys, application, print_on_screen, color
from rallyrobopilot.raycast_sensor import MAX_RAYCAST_DIST

from rallyrobopilot import prepare_game_app
from checkpoints import CheckpointManager


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class LinearPolicy:
    def __init__(self, n_inputs: int, genes: List[float]):
        self.n_inputs = n_inputs
        self.out_dim = 4 
        expected = self.out_dim * self.n_inputs + self.out_dim
        if len(genes) != expected:
            raise ValueError(f"Expected {expected} genes, got {len(genes)}")

        w_flat = genes[: self.out_dim * self.n_inputs]
        b_flat = genes[self.out_dim * self.n_inputs:]

        self.W = [w_flat[i * self.n_inputs:(i + 1) * self.n_inputs] for i in range(self.out_dim)]
        self.b = b_flat

    def __call__(self, x: List[float]) -> Tuple[float, float, float, float]:
        assert len(x) == self.n_inputs
        logits = []
        for o in range(self.out_dim):
            s = self.b[o]
            row = self.W[o]
            for j in range(self.n_inputs):
                s += row[j] * x[j]
            logits.append(s)
        probs = [math.tanh(v) for v in logits]
        probs_01 = [(p + 1.0) * 0.5 for p in probs]
        return tuple(probs_01)  # type: ignore 

class PolicyRunner(Entity):
    def __init__(
        self,
        car,
        cp_manager: CheckpointManager,
        genes: List[float],
        episode_time: float = 35.0,
        idle_timeout: float = 6.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.car = car
        self.cp_manager = cp_manager
        self.episode_time = episode_time
        self.idle_timeout = idle_timeout

        # Inputs = raycasts + speed_norm
        self.n_rays = getattr(car.multiray_sensor, "nbr_ray", 15)
        self.n_inputs = self.n_rays + 1
        self.policy = LinearPolicy(self.n_inputs, genes)

        # progression stats
        self.best_cp_idx = 0
        self.best_lap = 0

        self.total_speed = 0.0
        self.frame_count = 0

        #Colisions
        self.prev_hitting_wall = False
        self.last_collision_pos = None
        self.collision_min_dist = 3.0 
        self.collision_penalty = 0.0

        # time
        self.episode_start_t = time.time()
        self.last_move_t = self.episode_start_t
        self.alive = True

        self._hook_checkpoints()

        print_on_screen("[Worker] Episode started", position=(-0.9, 0.45), scale=0.8, duration=2) # type: ignore

    def _hook_checkpoints(self):
        orig_on_cp = self.cp_manager.on_checkpoint_passed
        orig_on_lap = self.cp_manager.on_lap_completed

        def on_cp(idx, cp):
            self.best_cp_idx = max(self.best_cp_idx, idx)
            orig_on_cp(idx, cp)

        def on_lap(lap):
            self.best_lap = max(self.best_lap, lap)
            orig_on_lap(lap)

        self.cp_manager.on_checkpoint_passed = on_cp
        self.cp_manager.on_lap_completed = on_lap


    def _clear_inputs(self):
        for k in ["w", "a", "s", "d", "up arrow", "down arrow", "left arrow", "right arrow"]:
            held_keys[k] = 0 # type: ignore

    def _get_observation(self) -> List[float]:
        mrs = getattr(self.car, "multiray_sensor", None)
        if mrs is not None:
            dists = mrs.collect_sensor_values()
        else:
            dists = []

        if not dists:
            dists = [MAX_RAYCAST_DIST] * self.n_rays
        else:
            if len(dists) < self.n_rays:
                dists = list(dists) + [MAX_RAYCAST_DIST] * (self.n_rays - len(dists))
            else:
                dists = list(dists[: self.n_rays])

        normed = [clamp(d / MAX_RAYCAST_DIST, 0.0, 1.0) for d in dists]

        v = float(getattr(self.car, "speed", 0.0))
        car_topspeed = getattr(self.car, "topspeed", 30.0)
        v_norm = clamp(abs(v) / car_topspeed, 0.0, 1.0)

        self.total_speed += abs(v)
        self.frame_count += 1

        normed.append(v_norm)
        return normed

    def _apply_actions(self, pf: float, pb: float, pl: float, pr: float):
        f = pf > 0.5
        b = pb > 0.5
        l = pl > 0.5
        r = pr > 0.5

        if f and b: 
            if pf > pb:
                b = False
            else:
                f = False

        if l and r: 
            if pl > pr:
                r = False
            else:
                l = False

        self._clear_inputs()
        if f:
            held_keys["w"] = 1  # type: ignore
        if b:
            held_keys["s"] = 1  # type: ignore
        if l:
            held_keys["a"] = 1  # type: ignore
        if r:
            held_keys["d"] = 1  # type: ignore

    def _finish(self):
        now = time.time()
        elapsed = now - self.episode_start_t

        PROGRESSION_REWARD = 20000.0
        base = self.best_lap * PROGRESSION_REWARD * len(self.cp_manager.checkpoints) + self.best_cp_idx * PROGRESSION_REWARD

        COLLISION_PENALTY_FACTOR = PROGRESSION_REWARD / 5.0
        collision_penalty_total = self.collision_penalty * COLLISION_PENALTY_FACTOR

        avg_speed = (self.total_speed / self.frame_count) if self.frame_count > 0 else 0
        
        SPEED_REWARD = 100.0
        speed_bonus = avg_speed * SPEED_REWARD

        if self.best_cp_idx == 0 and elapsed < self.episode_time:
             IDLE_PENALTY = 50000.0
             fitness = base - collision_penalty_total - IDLE_PENALTY + speed_bonus
        else:
             fitness = base - collision_penalty_total + speed_bonus

        print(f"FITNESS: {fitness}", flush=True)
        print(f"[Worker] Finished episode: fitness={fitness:.2f}, laps={self.best_lap}, cp={self.best_cp_idx}, col_pen={self.collision_penalty:.1f}, t={elapsed:.1f}s")

        self._clear_inputs()
        self.alive = False
        application.quit()

    def update(self):
        if not self.alive:
            return

        now = time.time()
        elapsed = now - self.episode_start_t

        if elapsed > self.episode_time:
            self._finish()
            return

        v = abs(float(getattr(self.car, "speed", 0.0)))
        if v > 0.5:
            self.last_move_t = now
        elif (now - self.last_move_t) > self.idle_timeout:
            self._finish()
            return

        # Colision detection
        hit = getattr(self.car, "hitting_wall", False)

        if hit and not self.prev_hitting_wall:
            pos = self.car.world_position
            px, pz = float(pos.x), float(pos.z)

            if self.last_collision_pos is None:
                self.collision_penalty += 1.0
                self.last_collision_pos = (px, pz)
            else:
                lx, lz = self.last_collision_pos
                dx = px - lx
                dz = pz - lz
                dist2 = dx*dx + dz*dz

                if dist2 > (self.collision_min_dist ** 2):
                    self.collision_penalty += 1.0
                    self.last_collision_pos = (px, pz)

        self.prev_hitting_wall = hit

        obs = self._get_observation()
        pf, pb, pl, pr = self.policy(obs)
        self._apply_actions(pf, pb, pl, pr)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--track", type=str, default="SimpleTrack")
    p.add_argument("--genes", type=str, required=True, help="comma-separated floats")
    p.add_argument("--episode_time", type=float, default=35.0)
    p.add_argument("--idle_timeout", type=float, default=6.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    genes = [float(x) for x in args.genes.split(",") if x.strip()]

    app, car = prepare_game_app(args.track)

    cp_manager = CheckpointManager(
        car=car,
        track_name=args.track,
        save_dir="tracks",
    )

    runner = PolicyRunner(
        car=car,
        cp_manager=cp_manager,
        genes=genes,
        episode_time=args.episode_time,
        idle_timeout=args.idle_timeout,
        color=color.clear,
    )

    app.run()
