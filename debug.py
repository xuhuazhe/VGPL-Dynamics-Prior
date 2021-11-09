from plb.engine.taichi_env import TaichiEnv
from plb.config import load

import tqdm
import numpy as np


if __name__ == '__main__':
    cfg = load('../PlasticineLab/plb/envs/gripper.yml')
    env = TaichiEnv(cfg=cfg)
    for i in tqdm.trange(200 * 90 * 3):
        env.step(np.zeros(12))