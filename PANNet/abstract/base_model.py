import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """Forward pass logic
            :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """Model prints with number of trainable parameters
        """
        params = sum([param.numel() for param in self.parameters() if param.requires_grad])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
