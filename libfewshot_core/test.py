# -*- coding: utf-8 -*-
import os
import builtins
from logging import getLogger
from time import time



import numpy as np
import torch
from torch import nn
import torch.distributed as dist

import libfewshot_core.model as arch
from libfewshot_core.data import get_dataloader

from libfewshot_core.audio_augmentations import *

from libfewshot_core.utils import (
    init_logger_config,
    prepare_device,
    init_seed,
    create_dirs,
    AverageMeter,
    count_parameters,
    ModelType,
    TensorboardWriter,
    mean_confidence_interval,
    get_instance,
)
from .data.collates import get_mean_std

def map_q_to_s_runs(s: np.ndarray, r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Map query labels q onto the binary vector s, using run lengths r.
    Works even if s is all True, or r contains a mix of 1s and >1s.

    Parameters
    ----------
    s : np.ndarray, bool
        Binary vector containing True entries (query positions)
    r : np.ndarray, int
        Length of each consecutive True-run
    q : np.ndarray, bool
        Label per run (True=bad, False=good)

    Returns
    -------
    mapped_q : np.ndarray, bool
        Same shape as s, with True entries labeled according to q
    """
    s = np.asarray(s, dtype=bool)
    r = np.asarray(r, dtype=int)
    q = np.asarray(q, dtype=bool)

    if r.sum() != s.sum():
        raise ValueError(f"Sum of r ({r.sum()}) must equal number of True in s ({s.sum()})")
    if len(r) != len(q):
        raise ValueError(f"Length of r ({len(r)}) must equal length of q ({len(q)})")

    mapped_q = np.full_like(s, False, dtype=bool)
    true_idx = np.flatnonzero(s)

    # split true_idx according to r
    start = 0
    for run_idx, run_length in enumerate(r):
        end = start + run_length
        indices = true_idx[start:end]
        mapped_q[indices] = q[run_idx]
        start = end

    return mapped_q

def augment_images_with_mask(images, repeats, is_query_mask, mask, augmentation_fn, num_augmentations=10):
    """
    Augment images based on repeats vector and binary mask.
    
    Args:
        images (torch.Tensor): Input images tensor of shape [N, C, H, W] or [N, ...]
        repeats (list or torch.Tensor): Vector of positive integers (minimum value 1)
        mask (torch.Tensor): Binary mask of shape [len(repeats)]
        augmentation_fn (callable): Function that takes an image and returns num_augmentations augmented images
        num_augmentations (int): Number of augmented images to generate per masked image (M)
    
    Returns:
        torch.Tensor: Augmented images tensor with increased size
    """
    # Convert repeats to numpy for easier indexing
    if isinstance(repeats, torch.Tensor):
        repeats_np = repeats.cpu().numpy()
    else:
        repeats_np = np.array(repeats)
    
    # Convert mask to numpy
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)
    
    # Calculate cumulative sum of repeats to find image indices
    cumsum_repeats = np.concatenate([[0], np.cumsum(repeats_np)])
    
    # Create list to hold all images (original + augmented)
    augmented_images_list = []
    current_idx = 0
    in_batch_ood_queries = map_q_to_s_runs(s=is_query_mask, r=repeats_np, q=mask_np)
    for i in range(len(images)):
        if in_batch_ood_queries[i]:
            for _ in range(num_augmentations):
                # This image needs augmentation
                original_img = images[i].clone()
                # Apply augmentation to get M augmented images
                augmented_batch = augmentation_fn(original_img.unsqueeze(0))
                # Add all augmented images to the list
                if isinstance(augmented_batch, torch.Tensor):
                    for aug_img in augmented_batch:
                        augmented_images_list.append(aug_img)
                else:
                    augmented_images_list.extend(augmented_batch)
        else:
            # No augmentation needed, just copy original image
            augmented_images_list.append(images[i])
    
    # for i in range(len(repeats_np)):
    #     start_idx = cumsum_repeats[i]
    #     end_idx = cumsum_repeats[i + 1]
    #     num_images_in_range = end_idx - start_idx
        
    #     if mask_np[i]:
    #         # This range needs augmentation
    #         for img_idx in range(start_idx, end_idx):
    #             # Get the original image
    #             original_img = images[img_idx]
                
    #             # Apply augmentation to get M augmented images
    #             augmented_batch = augmentation_fn(original_img.unsqueeze(0))
                
    #             # Add all augmented images to the list
    #             if isinstance(augmented_batch, torch.Tensor):
    #                 for aug_img in augmented_batch:
    #                     augmented_images_list.append(aug_img)
    #             else:
    #                 augmented_images_list.extend(augmented_batch)
    #     else:
    #         # No augmentation needed, just copy original images
    #         for img_idx in range(start_idx, end_idx):
    #             augmented_images_list.append(images[img_idx])
    
    # Stack all images into a single tensor
    augmented_images = torch.stack(augmented_images_list, dim=0)
    
    return augmented_images


class Test(object):
    """
    The tester.

    Build a tester from config dict, set up model from a saved checkpoint, etc. Test and log.
    """

    def __init__(self, rank, config, result_path=None):
        self.rank = rank
        self.config = config
        self.config["rank"] = rank
        self.result_path = result_path
        self.distribute = self.config["n_gpu"] > 1
        self.viz_path, self.state_dict_path = self._init_files(config)
        self.logger = self._init_logger()
        self.device, self.list_ids = self._init_device(rank, config)
        self.writer = self._init_writer(self.viz_path)
        self.test_meter = self._init_meter()
        print(config)
        self.model, self.model_type = self._init_model(config)
        # For Jerry -- Add also validation loader (also change yaml accordingly to account for val episodes)
        self.val_loader = self._init_val_dataloader(config)
        
        self.test_loader = self._init_dataloader(config)

    def test_loop(self):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """
        total_accuracy = 0.0
        total_h = np.zeros(self.config["test_epoch"])
        total_accuracy_vector = []
        
        self._validate(0, data_loader=self.val_loader, val_test_flag = 'val', enhance_classification_via_energy=False, update_threshold=True)  # validate to set the uncertainty threshold
        

        val_test_flag = 'test'
        for epoch_idx in range(self.config["test_epoch"]):
            print("============ Testing on the test set ============")
            # _, accuracies = self._validate(epoch_idx, update_threshold=True)  # this needs to be fixed (val loader only)
            # this is for test loader where you will have to put also the augmentations where you will save the entire batch containing the supports with the queries that the model is uncertain about
            _, accuracies = self._validate(epoch_idx, data_loader=self.test_loader, val_test_flag = val_test_flag, enhance_classification_via_energy=False, num_augmentations=10)  
            # accuracies is a list of scalars or tensors; convert to numpy array
            if len(accuracies) == 0:
                arr = np.array([])
            else:
                first = accuracies[0]
                if isinstance(first, torch.Tensor):
                    try:
                        arr = torch.stack([a.detach().cpu().float().squeeze() for a in accuracies]).numpy()
                    except Exception:
                        # fallback: convert elementwise
                        arr = np.array([float(a.item()) if isinstance(a, torch.Tensor) else float(a) for a in accuracies])
                else:
                    arr = np.array(accuracies, dtype=float)
            test_accuracy, h = mean_confidence_interval(arr)
            print("Test Accuracy: {:.3f}\t h: {:.3f}".format(test_accuracy, h))
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[epoch_idx] = h

        # total_accuracy_vector may contain tensors or floats; convert to numeric array
        if len(total_accuracy_vector) == 0:
            avg_arr = np.array([])
        else:
            first = total_accuracy_vector[0]
            if isinstance(first, torch.Tensor):
                try:
                    avg_arr = torch.stack([a.detach().cpu().float().squeeze() for a in total_accuracy_vector]).numpy()
                except Exception:
                    avg_arr = np.array([float(a.item()) if isinstance(a, torch.Tensor) else float(a) for a in total_accuracy_vector])
            else:
                avg_arr = np.array(total_accuracy_vector, dtype=float)

        aver_accuracy, h = mean_confidence_interval(avg_arr)
        print("Aver Accuracy: {:.3f}\t Aver h: {:.3f}".format(aver_accuracy, h))
        print("............Testing is end............")

        if self.writer is not None:
            self.writer.close()
            if self.distribute:
                dist.barrier()
        elif self.distribute:
            dist.barrier()

    # def _validate(self, epoch_idx, **kwargs):
    #     """
    #     The test stage.

    #     Args:
    #         epoch_idx (int): Epoch index.

    #     Returns:
    #         float: Acc.
    #     """
    #     # switch to evaluate mode
    #     self.model.eval()
    #     if self.distribute:
    #         self.model.module.reverse_setting_info()
    #     else:
    #         self.model.reverse_setting_info()
    #     meter = self.test_meter
    #     meter.reset()
    #     episode_size = self.config["episode_size"]
    #     accuracies = []

    #     end = time()
    #     enable_grad = self.model_type != ModelType.METRIC
    #     log_scale = self.config["episode_size"]
    #     with torch.set_grad_enabled(enable_grad):
    #         loader = self.test_loader
    #         for batch_idx, batch in enumerate(zip(*loader)):
    #             if self.rank == 0:
    #                 self.writer.set_step(
    #                     int(
    #                         (
    #                             epoch_idx * len(self.test_loader)
    #                             + batch_idx * episode_size
    #                         )
    #                         * self.config["tb_scale"]
    #                     )
    #                 )

    #             meter.update("data_time", time() - end)

    #             # calculate the output
    #             calc_begin = time()
    #             output, acc = self.model([elem for each_batch in batch for elem in each_batch], update_threshold=kwargs.get('update_threshold', False))
    #             # print(output.shape)
    #             # input()
    #             accuracies.append(acc)
    #             meter.update("calc_time", time() - calc_begin)

    #             # measure accuracy and record loss
    #             meter.update("acc", acc)

    #             # measure elapsed time
    #             meter.update("batch_time", time() - end)

    #             if ((batch_idx + 1) * log_scale % self.config["log_interval"] == 0) or (
    #                 batch_idx + 1
    #             ) * episode_size >= max(map(len, loader)) * log_scale:
    #                 info_str = (
    #                     "Epoch-({}): [{}/{}]\t"
    #                     "Time {:.3f} ({:.3f})\t"
    #                     "Calc {:.3f} ({:.3f})\t"
    #                     "Data {:.3f} ({:.3f})\t"
    #                     "Acc@1 {:.3f} ({:.3f})".format(
    #                         epoch_idx,
    #                         (batch_idx + 1) * log_scale,
    #                         max(map(len, loader)) * log_scale,
    #                         meter.last("batch_time"),
    #                         meter.avg("batch_time"),
    #                         meter.last("calc_time"),
    #                         meter.avg("calc_time"),
    #                         meter.last("data_time"),
    #                         meter.avg("data_time"),
    #                         meter.last("acc"),
    #                         meter.avg("acc"),
    #                     )
    #                 )
    #                 print(info_str)
    #             end = time()

    #     if kwargs.get('update_threshold', False):
    #         if self.distribute:
    #             self.model.module.get_uncertainty_threshold(policy='mean')
    #         else:
    #             self.model.get_uncertainty_threshold(policy='mean')

    #     if self.distribute:
    #         self.model.module.reverse_setting_info()
    #     else:
    #         self.model.reverse_setting_info()
    #     return meter.avg("acc"), accuracies
    
    def _validate(self, epoch_idx, data_loader, val_test_flag='val', **kwargs):
        """
        The test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        
        # switch to evaluate mode
        self.model.eval()
        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        meter = self.test_meter
        meter.reset()
        episode_size = self.config["episode_size"]
        accuracies = []

        end = time()
        enable_grad = self.model_type != ModelType.METRIC
        log_scale = self.config["episode_size"]
        
        # Import audio augmentations if needed
        enhance_classification = kwargs.get('enhance_classification_via_energy', False)
        
        with torch.set_grad_enabled(enable_grad):
            loader = data_loader
            for batch_idx, batch in enumerate(zip(*loader)):
                if self.rank == 0:
                    self.writer.set_step(
                        int(
                            (
                                epoch_idx * len(data_loader)
                                + batch_idx * episode_size
                            )
                            * self.config["tb_scale"]
                        )
                    )

                meter.update("data_time", time() - end)

                # calculate the output
                calc_begin = time()
                if val_test_flag == 'val' or enhance_classification==False:
                    # print('in val')
                    output, acc = self.model([elem for each_batch in batch for elem in each_batch], update_threshold=kwargs.get('update_threshold', True), enhance_classification_via_energy=kwargs.get('enhance_classification_via_energy', False))
                else:
                    _, acc, _, ood_mask, query_mask = self.model([elem for each_batch in batch for elem in each_batch], update_threshold=False, enhance_classification_via_energy=kwargs.get('enhance_classification_via_energy', True))
                    # print(ood_mask.sum())
                    # print('prior to augmentations accuracy is:{}'.format(acc.item()))
                    # print('applying augmentations to {} samples'.format(ood_mask))
                    # print('query mask sum is:{}'.format(query_mask))
                    # print('ood mask sum is:{}'.format(ood_mask.sum()))
                    # print('query mask is:{}'.format(query_mask.sum()))

                    if ood_mask.sum() > 0:
                        idxs = np.where(ood_mask)[0]
                        tmp = [elem for each_batch in batch for elem in each_batch]
                        image, global_target, repeats, support_size = tmp
                        # print('repeats are:{}'.format(repeats))
                        # print('shape of image is:{}'.format(image.shape))
                        
                        # MEAN,STD=get_mean_std(config, mode, modality)
                        temp = np.load('./Auxiliary/Clean_Mean_Std.npy')
                        MEAN, STD = temp.flatten().tolist()
                        # MEAN = [MEAN]
                        # STD = [STD]
                        audio_aug = lambda x: augment_spectrogram(x, mean=MEAN, std=STD, augmentation_type='noise_suppression')
                        augmented_images = augment_images_with_mask(
                            images=image,
                            repeats=repeats,
                            is_query_mask=query_mask,
                            mask=ood_mask,
                            augmentation_fn=audio_aug,
                            num_augmentations=kwargs['num_augmentations']
                        )
                        
                        repeats[idxs] += (kwargs['num_augmentations'] - 1)
                        # print('shape of augmented_images is:{}'.format(augmented_images.shape))
                        
                        # print('repeats are:{}'.format(repeats))
                        
                        # print('repeats sum are:{}'.format(repeats.sum()))
                        
                        
                        _, acc, _, _, _ = self.model([augmented_images, global_target, repeats, support_size], update_threshold=kwargs.get('update_threshold', False), enhance_classification_via_energy=kwargs.get('enhance_classification_via_energy', False))
                        
                        # print('after augmentations accuracy is:{}'.format(acc.item()))
                        
                        
                # Apply augmentations for uncertain predictions
                if False:
                    # Get uncertainty scores from model
                    if self.distribute:
                        uncertainty_mask = self.model.module.get_uncertain_samples(output)
                    else:
                        uncertainty_mask = self.model.get_uncertain_samples(output)
                    
                    # If there are uncertain samples, apply augmentations
                    if uncertainty_mask.any():
                        # Extract uncertain features
                        batch_list = [elem for each_batch in batch for elem in each_batch]
                        uncertain_features = [batch_list[i] for i in range(len(batch_list)) if uncertainty_mask[i]]
                        
                        # Apply audio augmentations
                        augmented_features = []
                        for feature in uncertain_features:
                            aug_feature = audio_aug(feature)
                            augmented_features.append(aug_feature)
                        
                        # Re-run model on augmented features
                        if augmented_features:
                            aug_output, aug_acc = self.model(augmented_features, update_threshold=False)
                            # Update output and accuracy for uncertain samples
                            # This is a simplified approach - you may want to ensemble or average
                            acc = (acc + aug_acc) / 2.0
                
                accuracies.append(acc)
                meter.update("calc_time", time() - calc_begin)

                # measure accuracy and record loss
                meter.update("acc", acc)

                # measure elapsed time
                meter.update("batch_time", time() - end)

                if ((batch_idx + 1) * log_scale % self.config["log_interval"] == 0) or (
                    batch_idx + 1
                ) * episode_size >= max(map(len, loader)) * log_scale:
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Calc {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@1 {:.3f} ({:.3f})".format(
                            epoch_idx,
                            (batch_idx + 1) * log_scale,
                            max(map(len, loader)) * log_scale,
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("calc_time"),
                            meter.avg("calc_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc"),
                            meter.avg("acc"),
                        )
                    )
                    print(info_str)
                end = time()
            # self.model.get_uncertainty_threshold(policy='mean', normalize=True)
            

        if kwargs.get('update_threshold', False):
            if self.distribute:
                self.model.module.get_uncertainty_threshold(policy='mean')
            else:
                print('getting uncertainty threshold')
                self.model.get_uncertainty_threshold(policy='mean')
                # exit()

        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        return meter.avg("acc"), accuracies

    def _init_files(self, config):
        """
        Init result_path(log_path, viz_path) from the config dict.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (viz_path, checkpoints_path).
        """
        if self.result_path is not None:
            result_path = self.result_path
        else:
            result_dir = "{}-{}-{}-{}-{}".format(
                config["classifier"]["name"],
                # you should ensure that data_root name contains its true name
                config["data_root"].split("/")[-1],
                config["backbone"]["name"],
                config["way_num"],
                config["shot_num"],
            )
            result_path = os.path.join(config["result_root"], result_dir)
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")

        init_logger_config(
            config["log_level"],
            log_path,
            config["classifier"]["name"],
            config["backbone"]["name"],
            is_train=False,
            rank=self.rank,
        )

        state_dict_path = os.path.join(result_path, "checkpoints", "model_best.pth")
        if self.rank == 0:
            create_dirs([result_path, log_path, viz_path])
            

        return viz_path, state_dict_path

    def _init_logger(self):
        self.logger = getLogger(__name__)

        # Hack print
        def use_logger(msg, level="info"):
            if self.rank != 0:
                return
            if level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            else:
                raise ("Not implemente {} level log".format(level))

        builtins.print = use_logger

        return self.logger

    def _init_dataloader(self, config):
        """
        Init the Test dataloader.

        Args:
            config (dict): Parsed config file.

        Returns:
            Dataloader: Test_loader.
        """
        self._check_data_config()
        distribute = self.distribute
        test_loader = get_dataloader(config, "test", self.model_type, distribute)

        return test_loader
    
    def _init_val_dataloader(self, config):
        """
        Init the Validation dataloader.

        Args:
            config (dict): Parsed config file.

        Returns:
            Dataloader: Val_loader.
        """
        self._check_data_config()
        distribute = self.distribute
        val_loader = get_dataloader(config, "val", self.model_type, distribute)

        return val_loader

    def _check_data_config(self):
        """
        Check the config params.
        """
        # check: episode_size >= n_gpu and episode_size != 0
        assert (
            self.config["episode_size"] >= self.config["n_gpu"]
            and self.config["episode_size"] != 0
        ), "episode_size {} should be >= n_gpu {} and != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_size % n_gpu == 0
        assert (
            self.config["episode_size"] % self.config["n_gpu"] == 0
        ), "episode_size {} % n_gpu {} != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_num % episode_size == 0
        assert (
            self.config["train_episode"] % self.config["episode_size"] == 0
        ), "train_episode {} % episode_size  {} != 0".format(
            self.config["train_episode"], self.config["episode_size"]
        )

        assert (
            self.config["test_episode"] % self.config["episode_size"] == 0
        ), "test_episode {} % episode_size  {} != 0".format(
            self.config["test_episode"], self.config["episode_size"]
        )

    def _init_model(self, config):
        """
        Init model (backbone+classifier) from the config dict and load the best checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        emb_func = get_instance(arch, "backbone", config)
        model_kwargs = {
            "way_num": config["way_num"],
            "shot_num": config["shot_num"] * config["augment_times"],
            "query_num": config["query_num"],
            "test_way": config["test_way"],
            "test_shot": config["test_shot"] * config["augment_times"],
            "test_query": config["test_query"],
            "emb_func": emb_func,
            "device": self.device,
        }
        model = get_instance(arch, "classifier", config, **model_kwargs)

        print(model)
        print("Trainable params in the model: {}.".format(count_parameters(model)))
        print("Loading the state dict from {}.".format(self.state_dict_path))
        state_dict = torch.load(self.state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

        if self.distribute:
            # higher order grad of BN in multi gpu will conflict with syncBN
            # FIXME MAML with multi GPU is conflict with syncBN
            if not (
                self.config["classifier"]["name"] in ["MAML"]
                and self.config["n_gpu"] > 1
            ):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                print(
                    "{} with multi GPU will conflict with syncBN".format(
                        self.config["classifier"]["name"]
                    ),
                    level="warning",
                )
            model = model.to(self.rank)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True,
            )

            return model, model.module.model_type
        else:
            model = model.to(self.device)

            return model, model.model_type

    def _init_device(self, rank, config):
        """
        Init the devices from the config file.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of devices and list_ids.
        """
        init_seed(config["seed"], config["deterministic"])
        device, list_ids = prepare_device(
            rank,
            config["device_ids"],
            config["n_gpu"],
            backend="nccl"
            if "dist_backend" not in self.config
            else self.config["dist_backend"],
            dist_url="tcp://127.0.0.1:" + str(config["port"])
            if "dist_url" not in self.config
            else self.config["dist_url"],
        )
        torch.cuda.set_device(self.rank)

        return device, list_ids

    def _init_meter(self):
        """
        Init the AverageMeter of test stage to cal avg... of batch_time, data_time, calc_time and acc.

        Returns:
            AverageMeter: Test_meter.
        """
        test_meter = AverageMeter(
            "test", ["batch_time", "data_time", "calc_time", "acc"], self.writer
        )

        return test_meter

    def _init_writer(self, viz_path):
        """
        Init the tensorboard writer.

        Return:
            writer: tensorboard writer
        """
        if self.rank == 0:
            writer = TensorboardWriter(viz_path)
            return writer
        else:
            return None
