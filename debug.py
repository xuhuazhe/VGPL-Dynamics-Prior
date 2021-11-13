from plb.engine.taichi_env import TaichiEnv
from plb.config import load

import tqdm
import numpy as np

def main():
    cfg = load('../PlasticineLab/plb/envs/gripper.yml')
    env = TaichiEnv(cfg=cfg)
    for i in tqdm.trange(200):
        env.step(np.zeros(12))

if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    with open("stats.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('tottime')
        stats.print_stats()