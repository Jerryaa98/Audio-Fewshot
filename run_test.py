# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from libfewshot_core.config import Config
from libfewshot_core import Test


# PATH = "/root/LibFewShot/results/DeepBDC-KOS_0.2_alpha_spec-resnet12Bdc-5-1-Oct-23-2025-16-57-23"

############## DEEPBDC EXPERIMENT PATHS ################

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

########################################################################

############## META BASELINE EXPERIMENT PATHS ################

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

########################################################################

############## R2D2 EXPERIMENT PATHS ################

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

############## PROTO NET EXPERIMENT PATHS ################

# Proto 5 shot iid seed 0
# PATH = "/root/LibFewShot/results/ProtoNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-15-25-30"

# Proto 5 shot iid seed 1
# PATH = "/root/LibFewShot/results/ProtoNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-17-11-50"

# Proto 5 shot iid seed 42
# PATH = "/root/LibFewShot/results/ProtoNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-18-43-55"

# Proto 5 shot ood seed 0
# PATH = "/root/LibFewShot/results/ProtoNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-15-25-18"

# Proto 5 shot ood seed 1
# PATH = "/root/LibFewShot/results/ProtoNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-17-14-28 "

# Proto 5 shot ood seed 42
# PATH = "/root/LibFewShot/results/ProtoNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-18-48-33"

################################################################

# path from a trained on 1 alpha to test on 0 alpha
# meta-baseline 5 shot ood seed 0
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-5-Oct-09-2025-01-21-48"
# deepbdc 5 shot ood seed 0
# PATH = "/root/LibFewShot/results/DeepBDC-KOS_1_alpha_spec-resnet12Bdc-5-5-Oct-09-2025-01-41-51"

# DN4 5 shot ood seed 0
# PATH = "/root/LibFewShot/results/DN4-KOS_1_alpha_spec-Conv64F-5-5-Oct-08-2025-04-22-24"


#---------------------------------#---------------------------------#---------------------------------



# 'Baseline': 
# PATH = "/root/LibFewShot/results/Baseline-KOS_1_alpha_spec-Conv64F-5-5-Oct-04-2025-15-53-49"
# 'Baseline++': 
# PATH = "/root/LibFewShot/results/BaselinePlus-KOS_1_alpha_spec-Conv64F-5-5-Oct-04-2025-17-34-20"
# 'Meta-Baseline': 
# PATH = "/root/LibFewShot/results/MetaBaseline-KOS_1_alpha_spec-Conv64F-5-5-Oct-09-2025-01-21-49"

# Meta-Learning family
# 'ANIL': 
# PATH = "/root/LibFewShot/results/ANIL-KOS_1_alpha_spec-Conv64F-5-5-Oct-03-2025-01-22-37"
# 'LEO': 
# PATH = "/root/LibFewShot/results/LEO-KOS_1_alpha_spec-Conv64F-5-5-Oct-05-2025-04-25-19"
# 'MAML': 
# PATH = "/root/LibFewShot/results/MAML-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-15-29-07"
# 'R2D2': 
# PATH = "/root/LibFewShot/results/R2D2-KOS_1_alpha_spec-Conv64F-5-5-Oct-05-2025-11-37-37"
# 'METAL': 
# PATH = "/root/LibFewShot/results/METAL-KOS_1_alpha_spec-Conv64F-5-5-Oct-05-2025-08-39-47"
# 'BOIL': 
# PATH = "/root/LibFewShot/results/BOIL-KOS_1_alpha_spec-Conv64F-5-5-Oct-04-2025-21-23-14"

# Metric-based family
# 'ADM': 
# PATH = "/root/LibFewShot/results/ADM-KOS_1_alpha_spec-Conv64F-5-5-Oct-07-2025-05-20-19"
# 'ATL-NET': 
# PATH = "/root/LibFewShot/results/ATLNet-KOS_1_alpha_spec-Conv64F-5-5-Dec-29-2025-13-16-15"
# 'DN4': 
# PATH = "/root/LibFewShot/results/DN4-KOS_1_alpha_spec-Conv64F-5-5-Oct-07-2025-18-10-51"
# 'ProtoNet': 
# PATH = "/root/LibFewShot/results/ProtoNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-24-2025-15-25-30"
# 'MCL': 
# PATH = "/root/LibFewShot/results/MCL-KOS_1_alpha_spec-Conv64F_MCL-5-5-Oct-07-2025-08-11-34"
# 'ADM_KL': 
PATH = "/root/LibFewShot/results/ADM_KL-KOS_1_alpha_spec-Conv64F-5-5-Oct-07-2025-22-47-55"
# 'RelationNet': 
# PATH = "/root/LibFewShot/results/RelationNet-KOS_1_alpha_spec-Conv64F-5-5-Oct-07-2025-10-42-09"


VAR_DICT = {
    "test_epoch": 1,
    "test_episode": 400
}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    print("Loaded config from {}".format(os.path.join(PATH, "config.yaml")))
    print(config)
    # input()
    # PATH = config["PATH"]

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
