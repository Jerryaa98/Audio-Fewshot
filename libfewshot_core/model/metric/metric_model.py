# -*- coding: utf-8 -*-
from abc import abstractmethod

from libfewshot_core.model.abstract_model import AbstractModel
from libfewshot_core.utils import ModelType


class MetricModel(AbstractModel):
    def __init__(self, init_type="normal", **kwargs):
        super(MetricModel, self).__init__(init_type, ModelType.METRIC, **kwargs)

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass
