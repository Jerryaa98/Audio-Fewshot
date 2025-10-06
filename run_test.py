# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "./results/Baseline-KOS_1_alpha_spec-Conv64F-5-1-Oct-04-2025-11-58-31"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 1000,
    "episode_size": 1,
    "modality": 'audio',
    'mean_std_file': './Auxiliary/Clean_Mean_Std.npy',
    'class_per_split': './Auxiliary/KOS_paper_splits.npy',
    'ood': False

}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
