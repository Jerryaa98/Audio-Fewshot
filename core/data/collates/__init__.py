# -*- coding: utf-8 -*-
from .collate_functions import GeneralCollateFunction, FewShotAugCollateFunction, FewShotAugCollateFunctionAudio
from .contrib import get_augment_method,get_mean_std
from ...utils import ModelType


def get_collate_function(config, trfms, mode, model_type, modality='image'):
    """Set the corresponding `collate_fn` by dict.

    + For finetuning-train, return `GeneralCollateFunction`
    + For finetuning-val, finetuning-test and meta/metric-train/val/test, return `FewShotAugCollateFunction`

    Args:
        config (dict): A LFS setting dict.
        trfms (list): A torchvision transform list.
        mode (str): Model mode in ['train', 'test', 'val']
        model_type (ModelType): An ModelType enum value of model.

    Returns:
        [type]: [description]
    """
    assert (
        model_type != ModelType.ABSTRACT
    ), "model_type should not be ModelType.ABSTRACT"

    if mode == "train" and model_type == ModelType.FINETUNING:
        if modality == 'image':
            collate_function = GeneralCollateFunction(trfms, config["augment_times"])
        else:
            collate_function = GeneralCollateFunction(trfms, config["augment_times"])
    else:
        if modality == 'image':
            collate_function = FewShotAugCollateFunction(
                trfms,
                config["augment_times"],
                config["augment_times_query"],
                config["way_num"] if mode == "train" else config["test_way"],
                config["shot_num"] if mode == "train" else config["test_shot"],
                config["query_num"] if mode == "train" else config["test_query"],
            )
        else:
            collate_function = FewShotAugCollateFunctionAudio(
                trfms,
                config["augment_times"],
                config["augment_times_query"],
                config["way_num"] if mode == "train" else config["test_way"],
                config["shot_num"] if mode == "train" else config["test_shot"],
                config["query_num"] if mode == "train" else config["test_query"],
                mode=mode,
            )


    return collate_function
