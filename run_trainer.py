# -*- coding: utf-8 -*-
import sys
import argparse

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer


parser_main = argparse.ArgumentParser(description="OurAudio")
parser_main.add_argument('--yaml_path', type=str, default="./config/anil.yaml")





def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    # config = Config("./config/anil.yaml").get_config_dict() - can run - running
    # config = Config("./config/maml.yaml").get_config_dict() - can run
    # config = Config("./config/boil.yaml").get_config_dict() - can run - running
    # config = Config("./config/leo.yaml").get_config_dict() - can run - running
    # config = Config("./config/metal.yaml").get_config_dict() - can run - running
    # config = Config("./config/r2d2.yaml").get_config_dict() - can run - running
    # config = Config("./config/versa.yaml").get_config_dict() - coudn't run
    # config = Config("./config/baseline.yaml").get_config_dict() - can run - running
    # config = Config("./config/baseline++.yaml").get_config_dict() - can run - running
    # config = Config("./config/deepbdc_pretrain.yaml").get_config_dict()
    # config = Config("./config/deepbdc.yaml").get_config_dict()
    # config = Config("./config/feat_pretrain.yaml").get_config_dict()
    # config = Config("./config/feat.yaml").get_config_dict()
    # config = Config("./config/frn_pretrain.yaml").get_config_dict()  # <--- need to adapt attention sizes
    # config = Config("./reproduce/MetaBaseline/MetaBaselinePretrain-KOS-resnet12.yaml").get_config_dict()
    # config = Config("./reproduce/MetaBaseline/MetaBaseline-KOS-resnet12-5-1.yaml").get_config_dict()
    # config = Config("./config/proto.yaml").get_config_dict() - can run - running
    args = parser_main.parse_args()
    config = Config(args.yaml_path).get_config_dict()


    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)