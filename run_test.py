# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "/root/LibFewShot/results/DeepBDC-KOS_0.2_alpha_spec-resnet12Bdc-5-1-Oct-23-2025-16-57-23"

# deepbdc 1shot iid seed 0
# PATH = "/root/LibFewShot/results/DeepBDC-KOS_1_alpha_spec-resnet12Bdc-5-1-Oct-09-2025-01-42-22"
# deepbdc 5shot iid seed 0
# PATH = "/root/LibFewShot/results/DeepBDC-KOS_1_alpha_spec-resnet12Bdc-5-5-Oct-09-2025-01-41-51"
# # deepbdc 1shot ood seed 0
# PATH = "/root/LibFewShot/results/DeepBDC-KOS_1_alpha_spec-resnet12Bdc-5-1-Oct-09-2025-01-42-22"
# # deepbdc 5shot ood seed 0
# PATH = "/root/LibFewShot/results/DeepBDC-KOS_1_alpha_spec-resnet12Bdc-5-5-Oct-09-2025-01-52-14"
# deepbdc 10 shot iid seed 1
# PATH = "/root/LibFewShot/results/DeepBDC-KOS_1_alpha_spec-resnet12Bdc-5-10-Oct-13-2025-14-55-38"
# deepbdc 10 shot ood seed 1
# PATH = "./results/DeepBDC-KOS_1_alpha_spec-resnet12Bdc-5-10-Oct-13-2025-20-27-58"

# meta baseline 1 shot iid seed 0
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-1-Oct-09-2025-01-20-49"
# meta baseline 1 shot ood seed 0
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-1-Oct-09-2025-01-23-50"
# meta baseline 5 shot iid seed 0
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-5-Oct-09-2025-01-21-49"
# meta baseline 5 shot ood seed 0
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-5-Oct-09-2025-01-21-48"
# meta baseline 10 shot iid seed 0
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-10-Oct-13-2025-05-31-40"
# meta baseline 10 shot ood seed 0
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-10-Oct-13-2025-06-58-35"

# r2d2 1 shot iid seed 0
# PATH = "/root/LibFewShot/results/R2D2-KOS_1_alpha_spec-Conv64F-5-1-Oct-05-2025-05-04-17"
# r2d2 1 shot ood seed 0
# PATH = "/root/LibFewShot/results/R2D2-KOS_1_alpha_spec-Conv64F-5-1-Oct-05-2025-18-05-51"
# r2d2 5 shot iid seed 0
# PATH = "/root/LibFewShot/results/R2D2-KOS_1_alpha_spec-Conv64F-5-5-Oct-05-2025-11-37-37"
# r2d2 5 shot ood seed 0
# PATH = "/root/LibFewShot/results/R2D2-KOS_1_alpha_spec-Conv64F-5-5-Oct-06-2025-14-30-33"
# r2d2 10 shot iid seed 0
# PATH = "/root/LibFewShot/results/R2D2-KOS_1_alpha_spec-Conv64F-5-10-Oct-13-2025-15-22-56"
# r2d2 10 shot ood seed 0
# PATH = "/root/LibFewShot/results/R2D2-KOS_1_alpha_spec-Conv64F-5-10-Oct-14-2025-00-44-35"



VAR_DICT = {
    "test_epoch": 1,
    "test_episode": 400
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
