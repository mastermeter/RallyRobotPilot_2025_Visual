# scripts/ga_master_parallel.py
import sys
import math
import time
import random
import subprocess
from dataclasses import dataclass
from typing import List

from concurrent.futures import ThreadPoolExecutor, as_completed


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class Individual:
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
    assert len(a) == len(b)
    if len(a) <= 1:
        return a[:]
    k = random.randint(1, len(a) - 1)
    return a[:k] + b[k:]


def eval_individual(
    genes: List[float],
    worker_script: str,
    track_name: str,
    episode_time: float,
    idle_timeout: float,
    idx: int,
    generation: int,
) -> float:
    genes_str = ",".join(f"{g:.6f}" for g in genes)

    cmd = [
        sys.executable,
        worker_script,
        f"--track={track_name}",
        f"--genes={genes_str}",
        f"--episode_time={episode_time}",
        f"--idle_timeout={idle_timeout}",
    ]

    print(f"[MASTER] Gen {generation}, indiv {idx}: launching worker...")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as e:
        print(f"[MASTER] ERROR launching worker for indiv {idx}: {e}")
        return -1e9

    if proc.stderr:
        print(f"[WORKER-{idx} STDERR]\n{proc.stderr}")

    fitness = -1e9
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("FITNESS:"):
            try:
                fitness = float(line.split(":", 1)[1].strip())
            except ValueError:
                fitness = -1e9
    print(f"[MASTER] Gen {generation}, indiv {idx}: fitness={fitness:.2f}")
    return fitness


def main():
    TRACK_NAME = "SimpleTrack"
    POP_SIZE = 60
    N_GENERATIONS = 300

    EPISODE_TIME = 60.0
    IDLE_TIMEOUT = 6.0

    MAX_WORKERS = 15 

    N_RAYS = 15
    N_INPUTS = N_RAYS + 1 #  rays + speed
    N_OUTPUTS = 4  
    N_PARAMS = N_OUTPUTS * N_INPUTS + N_OUTPUTS 

    WORKER_SCRIPT = "scripts/GA/ga_worker.py"

    population: List[Individual] = init_population(POP_SIZE, N_PARAMS, sigma=0.3)

    for gen in range(N_GENERATIONS):
        print(f"\n[MASTER] ===== Generation {gen} / {N_GENERATIONS} =====")
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for idx, indiv in enumerate(population):
                fut = executor.submit(
                    eval_individual,
                    indiv.genes,
                    WORKER_SCRIPT,
                    TRACK_NAME,
                    EPISODE_TIME,
                    IDLE_TIMEOUT,
                    idx,
                    gen,
                )
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                fitness = fut.result()
                population[idx].fitness = fitness

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        best = population[0]
        print(f"[MASTER] Gen {gen} best fitness: {best.fitness:.2f}")

        n_elite = max(1, POP_SIZE // 5)
        elites = population[:n_elite]

        new_pop: List[Individual] = []
        for e in elites:
            new_pop.append(Individual(genes=e.genes[:], fitness=-1e9))

        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(elites, 2)
            child_genes = crossover(p1.genes, p2.genes)
            child_genes = mutate(child_genes, sigma=0.1, p=0.1)
            new_pop.append(Individual(genes=child_genes, fitness=-1e9))

        population = new_pop

    population.sort(key=lambda ind: ind.fitness, reverse=True)
    best = population[0]
    print("\n[MASTER] === Training finished ===")
    print(f"Best fitness: {best.fitness:.2f}")
    print("Best genes:\n", ",".join(f"{g:.6f}" for g in best.genes))


if __name__ == "__main__":
    main()
