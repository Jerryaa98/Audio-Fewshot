# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch
from torch import nn

from libfewshot_core.utils import ModelType
from .init import init_weights
import numpy as np


def split_by_sizes(tensor, sizes, dim=0):
    """
    Split a tensor into a list of tensors along a given dimension 
    according to a vector of sizes.

    Parameters:
        tensor : torch.Tensor to split
        sizes  : list of integers (segment lengths)
        dim    : dimension along which to split

    Returns:
        list of torch.Tensors
    """
    if sum(sizes) != tensor.size(dim):
        raise ValueError("Sum of sizes must equal tensor dimension along dim")

    return list(torch.split(tensor, sizes, dim=dim))


def chunk_sum(tensor, chunk_size, dim=0):
    """
    Split tensor into non-overlapping consecutive chunks 
    of size `chunk_size` along dimension `dim`,
    sum each chunk, and return new tensor.
    """
    n = tensor.size(dim)
    if n % chunk_size != 0:
        raise ValueError("Tensor length along dim must be divisible by chunk_size")

    # reshape into (..., num_chunks, chunk_size, ...)
    shape = list(tensor.shape)
    num_chunks = n // chunk_size
    new_shape = shape[:dim] + [num_chunks, chunk_size] + shape[dim+1:]
    reshaped = tensor.reshape(*new_shape)

    # sum over the chunk dimension
    return reshaped.sum(dim=dim+1)


def hierarchical_cumsum(arr, num_first=4, num_second=5):
    """
    Split 1D array hierarchically and return cumsum of second-level sums as list of arrays.
    
    Parameters:
        arr         : 1D numpy array
        num_first   : number of chunks at first level
        num_second  : number of chunks at second level (per first-level chunk)
    
    Returns:
        list of np.ndarray: length num_first, each array of length num_second
    """
    arr = np.asarray(arr)
    N = arr.size
    
    if N % (num_first * num_second) != 0:
        raise ValueError(f"Length of array must be divisible by {num_first}*{num_second}")
    
    # First level
    level1_size = N // num_first
    level1 = arr.reshape(num_first, level1_size)
    
    # Second level
    level2_size = level1_size // num_second
    level2 = level1.reshape(num_first, num_second, level2_size)
    
    # Sum along last axis (second-level sublists)
    sums = level2.sum(axis=2)
    
    # Convert each inner list to np.array and take cumsum
    return ([np.cumsum(inner) for inner in sums], [np.array(inner) for inner in sums])


def hierarchical_cumsum_with_carry(arr, num_first=4, num_second=5):
    """
    Hierarchical split + cumulative sum + carry-over across first-level chunks.
    
    Parameters:
        arr         : 1D numpy array
        num_first   : number of first-level chunks
        num_second  : number of second-level chunks per first-level chunk
    
    Returns:
        list of np.ndarray: length num_first, each array of length num_second
    """
    arr = np.asarray(arr)
    N = arr.size
    
    if N % (num_first * num_second) != 0:
        raise ValueError(f"Length of array must be divisible by {num_first}*{num_second}")
    
    # First level
    level1_size = N // num_first
    level1 = arr.reshape(num_first, level1_size)
    
    # Second level
    level2_size = level1_size // num_second
    level2 = level1.reshape(num_first, num_second, level2_size)
    
    # Sum along last axis (second-level sublists)
    sums = level2.sum(axis=2)
    
    # Take cumsum within each inner array and add carry-over from previous
    result = []
    carry = 0
    for inner in sums:
        c = np.cumsum(inner) + carry
        carry = c[-1]  # last element becomes carry for next
        result.append(c)
    
    return result



class AbstractModel(nn.Module):
    def __init__(self, init_type, model_type=ModelType.ABSTRACT, **kwargs):
        super(AbstractModel, self).__init__()

        self.init_type = init_type
        self.model_type = model_type
        for key, value in kwargs.items():
            setattr(self, key, value)
        # print(self.is_clap)
        # input()

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass
    

    @abstractmethod 
    def get_uncertainty_threshold(self, policy='mean'):
        pass
    
    def forward(self, x, update_threshold=False, enhance_classification_via_energy=False):
        if self.training:
            return self.set_forward_loss(x)
        else:
            return self.set_forward(x)

    def train(self, mode=True):
        super(AbstractModel, self).train(mode)
        # for methods with distiller
        if hasattr(self, "distill_layer"):
            self.distill_layer.train(False)

    def eval(self):
        return super(AbstractModel, self).eval()

    def _init_network(self):
        init_weights(self, self.init_type)

    def _generate_local_targets(self, episode_size):
        local_targets = (
            torch.arange(self.way_num, dtype=torch.long)
            .view(1, -1, 1)
            .repeat(episode_size, 1, self.shot_num + self.query_num)
            .view(-1)
        )
        return local_targets

    def split_by_episode(self, features, mode, repeats=None, support_size=0):
        """
        split features by episode and
        generate local targets + split labels by episode
        """
        
        query_mask = None

        if repeats is not None:
            episode_size = (repeats.size(0) + support_size) // (self.way_num * (self.shot_num + self.query_num))
            temp = (self.way_num * (self.shot_num + self.query_num))
            size_per_n_way_per_episode = hierarchical_cumsum_with_carry(repeats.cpu().numpy(), episode_size, self.way_num)        
        else:
            episode_size = features.size(0) // (
                self.way_num * (self.shot_num + self.query_num)
            )
            
        
        if repeats is None:
            local_labels = (
                self._generate_local_targets(episode_size)
                .to(self.device)
                .contiguous()
                .view(episode_size, self.way_num, self.shot_num + self.query_num)
            )
        else:
            local_labels = (
                self._generate_local_targets(episode_size)
                .to(self.device)
                .contiguous()
                .view(episode_size, self.way_num, self.shot_num + self.query_num)
            )


        if mode == 1:  # input 2D, return 3D(with episode) E.g.ANIL & R2D2
            # print(features.shape)
            # input()
            if repeats is None:
                # print(features.shape)
                features = features.contiguous().view(
                    episode_size, self.way_num, self.shot_num + self.query_num, -1
                )
                # print(features.shape)
                # input()
                support_features = (
                    features[:, :, : self.shot_num, :]
                    .contiguous()
                    .view(episode_size, self.way_num * self.shot_num, -1)
                )
                query_features = (
                    features[:, :, self.shot_num :, :]
                    .contiguous()
                    .view(episode_size, self.way_num * self.query_num, -1)
                )
            else:
                query_mask = np.zeros(features.shape[0] ,dtype=bool)
                support_features = []
                query_features = []
                cnt = 0
                for i in range(episode_size):
                    query_features_temp = []
                    for j in range(len(size_per_n_way_per_episode[i])):
                        if j == 0:
                            if i == 0:
                                support_features.append(features[:self.shot_num])
                                query_features_temp.append(features[cnt+self.shot_num:cnt+self.shot_num+size_per_n_way_per_episode[i][j]])
                                query_mask[cnt+self.shot_num:cnt+self.shot_num+size_per_n_way_per_episode[i][j]] = True
                            else:
                                support_features.append(features[cnt+size_per_n_way_per_episode[i-1][-1]:cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num])
                                query_features_temp.append(features[cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num:size_per_n_way_per_episode[i][j] + cnt + self.shot_num])
                                query_mask[cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num:size_per_n_way_per_episode[i][j] + cnt + self.shot_num] = True
                        else:
                            support_features.append(features[cnt+size_per_n_way_per_episode[i][j-1]:cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num])
                            query_features_temp.append(features[cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num:cnt+size_per_n_way_per_episode[i][j] + self.shot_num])
                            query_mask[cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num:cnt+size_per_n_way_per_episode[i][j] + self.shot_num] = True
                        cnt += self.shot_num
                    query_features.append(torch.vstack(query_features_temp))
                
                support_features = (
                    torch.vstack(support_features)
                    .contiguous()
                    .view(episode_size, self.way_num * self.shot_num, -1)
                )
                # print('Support features: ', [x[0][0][-1] for x in support_features])
                # print('Query features: ', [x[0][0][-1] for x in query_features])
                # exit()


            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
            
            
                    

        elif mode == 2:  # input 4D, return 5D(with episode) E.g.DN4
            # print(features.shape)
            # input()
            b, c, h, w = features.shape
            if repeats is None:
                features = features.contiguous().view(
                    episode_size,
                    self.way_num,
                    self.shot_num + self.query_num,
                    c,
                    h,
                    w,
                )
                support_features = (
                    features[:, :, : self.shot_num, :, ...]
                    .contiguous()
                    .view(episode_size, self.way_num * self.shot_num, c, h, w)
                )
                query_features = (
                    features[:, :, self.shot_num :, :, ...]
                    .contiguous()
                    .view(episode_size, self.way_num * self.query_num, c, h, w)
                )
            else:
                query_mask = np.zeros(features.shape[0] ,dtype=bool)
                support_features = []
                query_features = []
                cnt = 0
                for i in range(episode_size):
                    query_features_temp = []
                    for j in range(len(size_per_n_way_per_episode[i])):
                        if j == 0:
                            if i == 0:
                                support_features.append(features[:self.shot_num])
                                query_features_temp.append(features[cnt+self.shot_num:cnt+self.shot_num+size_per_n_way_per_episode[i][j]])
                                query_mask[cnt+self.shot_num:cnt+self.shot_num+size_per_n_way_per_episode[i][j]] = True
                            else:
                                support_features.append(features[cnt+size_per_n_way_per_episode[i-1][-1]:cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num])
                                query_features_temp.append(features[cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num:size_per_n_way_per_episode[i][j] + cnt + self.shot_num])
                                query_mask[cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num:size_per_n_way_per_episode[i][j] + cnt + self.shot_num] = True
                        else:
                            support_features.append(features[cnt+size_per_n_way_per_episode[i][j-1]:cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num])
                            query_features_temp.append(features[cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num:cnt+size_per_n_way_per_episode[i][j] + self.shot_num])
                            query_mask[cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num:cnt+size_per_n_way_per_episode[i][j] + self.shot_num] = True
                        cnt += self.shot_num

                    query_features.append(torch.vstack(query_features_temp))
                
                support_features = (
                    torch.vstack(support_features)
                    .contiguous()
                    .view(episode_size, self.way_num * self.shot_num, c, h, w)
                )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif mode == 3:  # input 4D, return 4D(w/o episode) E.g.realationnet
            # print(episode_size)
            b, c, h, w = features.shape
            if repeats is None:
                
                features = features.contiguous().view(
                    self.way_num, self.shot_num + self.query_num, c, h, w
                )

                support_features = (
                    features[:, : self.shot_num, :, ...]
                    .contiguous()
                    .view(self.way_num * self.shot_num, c, h, w)
                )

                query_features = (
                    features[:, self.shot_num :, :, ...]
                    .contiguous()
                    .view(self.way_num * self.query_num, c, h, w)
                )
            else:
                query_mask = np.zeros(features.shape[0] ,dtype=bool)
                support_features = []
                query_features = []
                cnt = 0
                for i in range(episode_size):
                    query_features_temp = []
                    for j in range(len(size_per_n_way_per_episode[i])):
                        if j == 0:
                            if i == 0:
                                support_features.append(features[:self.shot_num])
                                query_features_temp.append(features[cnt+self.shot_num:cnt+self.shot_num+size_per_n_way_per_episode[i][j]])
                                query_mask[cnt+self.shot_num:cnt+self.shot_num+size_per_n_way_per_episode[i][j]] = True
                            else:
                                support_features.append(features[cnt+size_per_n_way_per_episode[i-1][-1]:cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num])
                                query_features_temp.append(features[cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num:size_per_n_way_per_episode[i][j] + cnt + self.shot_num])
                                query_mask[cnt+size_per_n_way_per_episode[i-1][-1] + self.shot_num:size_per_n_way_per_episode[i][j] + cnt + self.shot_num] = True
                        else:
                            support_features.append(features[cnt+size_per_n_way_per_episode[i][j-1]:cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num])
                            query_features_temp.append(features[cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num:cnt+size_per_n_way_per_episode[i][j] + self.shot_num])
                            query_mask[cnt+size_per_n_way_per_episode[i][j-1] + self.shot_num:cnt+size_per_n_way_per_episode[i][j] + self.shot_num] = True
                        cnt += self.shot_num
                    query_features.append(torch.vstack(query_features_temp))

            support_features = (
                    torch.vstack(support_features)
                    .contiguous()
                    .view(self.way_num * self.shot_num, c, h, w)
                )
            
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif (
            mode == 4
        ):  # finetuning baseline input 2D, return 2D(w/o episode) E.g.baseline set_forward
            features = features.view(self.way_num, self.shot_num + self.query_num, -1)
            support_features = (
                features[:, : self.shot_num, :]
                .contiguous()
                .view(self.way_num * self.shot_num, -1)
            )
            query_features = (
                features[:, self.shot_num :, :]
                .contiguous()
                .view(self.way_num * self.query_num, -1)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                self.way_num * self.query_num
            )
        else:
            raise Exception("mode should in [1,2,3,4], not {}".format(mode))

        return support_features, query_features, support_target, query_target, query_mask

    def reverse_setting_info(self):
        (
            self.way_num,
            self.shot_num,
            self.query_num,
            self.test_way,
            self.test_shot,
            self.test_query,
        ) = (
            self.test_way,
            self.test_shot,
            self.test_query,
            self.way_num,
            self.shot_num,
            self.query_num,
        )
