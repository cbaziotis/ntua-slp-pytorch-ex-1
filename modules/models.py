from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, **kwargs):
        """
        Define the layers and initialize them.

        Pytorch initializes the layers by default, with random weights,
        sampled from certain distribution. However, in some cases
        you might want to explicitly initialize some layers,
        either by sampling from a different distribution,
        or by using pretrained weights (word embeddings / transfer learning)

        Args:
        """
        super(BaselineModel, self).__init__()
        pass

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        pass
