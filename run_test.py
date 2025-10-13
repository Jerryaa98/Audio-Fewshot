# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "/root/LibFewShot/results/Baseline-KOS_1_alpha_spec-Conv64F-5-1-Oct-04-2025-11-58-31"
VAR_DICT = {
    "test_epoch": 1,
    "test_episode": 1000
}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    # print(config)
    # input()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
