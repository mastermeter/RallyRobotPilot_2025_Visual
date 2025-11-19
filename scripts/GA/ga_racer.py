# scripts/ga_racer.py
import os
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple

from ursina import Entity, held_keys, application, print_on_screen, color
from rallyrobopilot.raycast_sensor import MAX_RAYCAST_DIST

from rallyrobopilot import prepare_game_app   # ton game_launcher
from checkpoints import CheckpointManager


# =======================
#  Utilitaires GA
# =======================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class Individual:
    """Un individu = un simple réseau linéaire W,b pour décider [f, l, r]."""
    genes: List[float]
    fitness: float = -1e9


def init_population(pop_size: int, n_params: int, sigma: float = 0.5) -> List[Individual]:
    pop = []
    for _ in range(pop_size):
        genes = [random.gauss(0.0, sigma) for _ in range(n_params)]
        pop.append(Individual(genes=genes, fitness=-1e9))
    return pop


def mutate(genes: List[float], sigma: float = 0.1, p: float = 0.1) -> List[float]:
    out = []
    for g in genes:
        if random.random() < p:
            out.append(g + random.gauss(0.0, sigma))
        else:
            out.append(g)
    return out


def crossover(a: List[float], b: List[float]) -> List[float]:
    """Croisement 1 point."""
    assert len(a) == len(b)
    if len(a) <= 1:
        return a[:]
    k = random.randint(1, len(a) - 1)
    return a[:k] + b[k:]


# =======================
#  Policy linéaire
# =======================

class LinearPolicy:
    """
    x = [raycasts..., speed_norm] -> logits [f,l,r]

    On encode W,b dans un gros vecteur `genes`:
        W : out_dim x in_dim
        b : out_dim
    """

    def __init__(self, n_inputs: int, genes: List[float]):
        self.n_inputs = n_inputs
        self.out_dim = 3  # forward, left, right
        expected = self.out_dim * self.n_inputs + self.out_dim
        if len(genes) != expected:
            raise ValueError(f"Expected {expected} genes, got {len(genes)}")
        # découpage
        w_flat = genes[: self.out_dim * self.n_inputs]
        b_flat = genes[self.out_dim * self.n_inputs:]
        self.W = [w_flat[i * self.n_inputs:(i + 1) * self.n_inputs] for i in range(self.out_dim)]
        self.b = b_flat

    def __call__(self, x: List[float]) -> Tuple[float, float, float]:
        assert len(x) == self.n_inputs
        # logits = W x + b
        logits = []
        for o in range(self.out_dim):
            s = self.b[o]
            row = self.W[o]
            for j in range(self.n_inputs):
                s += row[j] * x[j]
            logits.append(s)
        # activation simple: tanh pour rester borné [-1,1]
        probs = [math.tanh(v) for v in logits]
        # On veut des trucs dans [0,1] pour threshold -> (tanh+1)/2
        probs_01 = [(p + 1.0) * 0.5 for p in probs]
        return tuple(probs_01)  # (pf, pl, pr)


# =======================
#  GA Controller
# =======================

class GeneticRacer(Entity):
    """
    Entity Ursina qui gère:
    - population GA
    - évaluation séquentielle des individus (un par un)
    - fitness = progression checkpoints - pénalités collisions - temps/immobilité
    """

    def __init__(
        self,
        car,
        cp_manager: CheckpointManager,
        pop_size: int = 20,
        n_generations: int = 50,
        episode_time: float = 35.0,
        idle_timeout: float = 5.0,
        track_name: str = "SimpleTrack",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.car = car
        self.cp_manager = cp_manager
        self.track_name = track_name

        #Colisions
        self.prev_hitting_wall = False
        self.last_collision_pos = None
        self.collision_min_dist = 3.0 

        # GA params
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.episode_time = episode_time     # durée max d'un essai (s)
        self.idle_timeout = idle_timeout     # si la voiture est (quasi) immobile trop longtemps -> stop

        # Inputs = raycasts + speed_norm
        self.n_rays = getattr(car.multiray_sensor, "nbr_ray", 15)
        self.n_inputs = self.n_rays + 1
        self.n_params = 3 * self.n_inputs + 3  # (W,b) du LinearPolicy

        # Population
        self.population: List[Individual] = init_population(self.pop_size, self.n_params, sigma=0.3)
        self.generation = 0
        self.current_idx = 0

        # episode state
        self.episode_start_t = 0.0
        self.last_move_t = 0.0
        self.best_cp_idx = 0
        self.best_lap = 0
        self.collision_penalty = 0.0
        self.alive = False

        # hook dans CheckpointManager pour suivre progression
        self._hook_checkpoints()

        print_on_screen(f"[GA] Pop={self.pop_size}, Params={self.n_params}", position=(-0.85, 0.45), scale=0.9, duration=3)

        # démarrer tout de suite
        self._start_episode()

    # ---------- Hooks sur les checkpoints ----------

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

    # ---------- Episode / carrière GA ----------

    def _reset_car(self):
        # Utilise la méthode de reset existante
        if hasattr(self.car, "reset_car"):
            self.car.reset_car()
        # Nettoie les vitesses
        self.car.speed = 0.0
        self.car.rotation_speed = 0.0

    def _clear_inputs(self):
        # remet les touches à 0
        for k in ["w", "a", "s", "d", "up arrow", "down arrow", "left arrow", "right arrow"]:
            held_keys[k] = 0

    def _start_episode(self):
        self._clear_inputs()
        self._reset_car()

        # Reset état checkpoints
        self.cp_manager.reset_race()

        # reset stats
        self.episode_start_t = time.time()
        self.last_move_t = self.episode_start_t
        self.best_cp_idx = 0
        self.best_lap = 0
        self.collision_penalty = 0.0
        self.alive = True

        self.prev_hitting_wall = False
        self.last_collision_pos = None

        indiv = self.population[self.current_idx]
        print(f"[GA] Gen {self.generation}, indiv {self.current_idx}/{self.pop_size}, genes_len={len(indiv.genes)}")

    def _end_episode(self):
        self.alive = False
        self._clear_inputs()

        now = time.time()
        elapsed = now - self.episode_start_t

        # fitness = nb CP + gros bonus par tour - pénalité collisions - petite pénalité temps
        indiv = self.population[self.current_idx]
        base = self.best_lap * 1000 + self.best_cp_idx
        time_pen = 0.1 * elapsed
        fitness = base - self.collision_penalty - time_pen

        indiv.fitness = fitness
        print(f"[GA] Finished indiv {self.current_idx}: fitness={fitness:.2f}, laps={self.best_lap}, cp={self.best_cp_idx}, col_pen={self.collision_penalty:.1f}, t={elapsed:.1f}s")

        # passe au suivant ou nouvelle génération
        self.current_idx += 1
        if self.current_idx >= self.pop_size:
            self._next_generation()
        else:
            self._start_episode()

    def _next_generation(self):
        # tri par fitness décroissant
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        best = self.population[0]
        print(f"[GA] === Generation {self.generation} done ===")
        print(f"     Best fitness: {best.fitness:.2f}")

        self.generation += 1
        if self.generation >= self.n_generations:
            print("[GA] Training finished, quitting.")
            application.quit()
            return

        # élitisme + reproduction simple
        n_elite = max(1, self.pop_size // 5)
        elites = self.population[:n_elite]

        new_pop: List[Individual] = []
        # garder les élites (clonés)
        for e in elites:
            new_pop.append(Individual(genes=e.genes[:], fitness=-1e9))

        # remplir le reste par crossover+mutation
        while len(new_pop) < self.pop_size:
            p1, p2 = random.sample(elites, 2)
            child_genes = crossover(p1.genes, p2.genes)
            child_genes = mutate(child_genes, sigma=0.1, p=0.1)
            new_pop.append(Individual(genes=child_genes, fitness=-1e9))

        self.population = new_pop
        self.current_idx = 0
        self._start_episode()

    # ---------- Politique et contrôle ----------

    def _get_observation(self) -> List[float]:
        # raycasts distances normalisées
        mrs = getattr(self.car, "multiray_sensor", None)
        if mrs is not None and hasattr(mrs, "distances"):
            dists = mrs.distances  # typiquement une liste de floats
        else:
            dists = []

        # fallback si pas de rayons
        if not dists:
            dists = [MAX_RAYCAST_DIST] * self.n_rays
        else:
            # tronquer/padder à n_rays
            if len(dists) < self.n_rays:
                dists = list(dists) + [MAX_RAYCAST_DIST] * (self.n_rays - len(dists))
            else:
                dists = list(dists[: self.n_rays])

        # normalisation [0,1]
        normed = [clamp(d / 60.0, 0.0, 1.0) for d in dists]  # 60m = max ~ piste

        # speed normalisée
        v = float(getattr(self.car, "speed", 0.0))
        v_norm = clamp(abs(v) / 30.0, 0.0, 1.0)

        normed.append(v_norm)
        return normed  # len == n_inputs

    def _apply_actions(self, pf: float, pl: float, pr: float):
        """
        pf, pl, pr ∈ [0,1] => bool sur touches.
        Seuils naïfs, et on évite de tourner gauche+droite en même temps.
        """
        f = pf > 0.5
        l = pl > 0.5
        r = pr > 0.5

        if l and r:
            # si conflit, garder le plus fort
            if pl > pr:
                r = False
            else:
                l = False

        # on reset tout puis on pose les touches
        self._clear_inputs()
        if f:
            held_keys["w"] = 1
        if l:
            held_keys["a"] = 1
        if r:
            held_keys["d"] = 1

    # ---------- Boucle update Ursina ----------

    def update(self):
        # Si fin du programme
        if self.generation >= self.n_generations:
            return

        # Si on attend rien (ex: pendant reset), on laisse passer un frame
        if not self.alive:
            return

        now = time.time()
        elapsed = now - self.episode_start_t

        # 1) Conditions d'arrêt épisode
        if elapsed > self.episode_time:
            self._end_episode()
            return

        # immobile trop longtemps ?
        v = abs(float(getattr(self.car, "speed", 0.0)))
        if v > 0.5:
            self.last_move_t = now
        elif (now - self.last_move_t) > self.idle_timeout:
            self._end_episode()
            return

        # pénalité collisions
        hit = getattr(self.car, "hitting_wall", False)

        if hit and not self.prev_hitting_wall:
            # frontière False -> True : début d'une collision
            pos = self.car.world_position
            px, pz = float(pos.x), float(pos.z)

            if self.last_collision_pos is None:
                # première collision de l'épisode
                self.collision_penalty += 1.0
                self.last_collision_pos = (px, pz)
            else:
                lx, lz = self.last_collision_pos
                dx = px - lx
                dz = pz - lz
                dist2 = dx*dx + dz*dz

                if dist2 > (self.collision_min_dist ** 2):
                    # nouvelle zone de collision
                    self.collision_penalty += 1.0
                    self.last_collision_pos = (px, pz)

        self.prev_hitting_wall = hit

        # 2) Observation + policy
        indiv = self.population[self.current_idx]
        policy = LinearPolicy(self.n_inputs, indiv.genes)
        obs = self._get_observation()
        pf, pl, pr = policy(obs)

        # 3) Appliquer actions
        self._apply_actions(pf, pl, pr)


# =======================
#  Entrée principale
# =======================

if __name__ == "__main__":
    TRACK_NAME = "SimpleTrack"   
    # NotSoSimpleTrack # SimpleTrack # SlightlyHarder # VisualTrack
    # VisualTrack/track_circuit2_metadata.json
    # VisualTrack/track_circuit3_metadata.json

    app, car = prepare_game_app(TRACK_NAME)

    cp_manager = CheckpointManager(
        car=car,
        track_name=TRACK_NAME,
        save_dir="tracks",
    )

    # Lancer le GA
    ga = GeneticRacer(
        car=car,
        cp_manager=cp_manager,
        pop_size=20,
        n_generations=30,
        episode_time=35.0,
        idle_timeout=6.0,
        track_name=TRACK_NAME,
        color=color.clear,   # juste pour éviter un cube visible
    )

    app.run()
