### Overview
This is you first assignment for familiarising yourself with PyTorch.

You have to complete the missing parts in the code, with the goal of training
a baseline RNN model for sentiment classification in Twitter messages.
The functions for loading the raw data (`utils/load_data.py`)
and the pretrained word embeddings (`utils/load_embeddings.py`)
are given to you.

### Key points
The key points of the first assignment are:
 - Utilize the dataloading abstractions of PyTorch,
    namely [torch.utils.data.Dataset](http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset)
    and [torch.utils.data.DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).
    Don't use torchtext. 
 - Initialize the embedding layer of your model with pretrained word embeddings.
 I recommend using [Glove's 50 dimensional vectors](http://nlp.stanford.edu/data/glove.twitter.27B.zip) ,
 as the performance of the model is irrelevant and using low-dimensional embeddings will speed things up.
 - Implement a baseline RNN model. Than means using the RNNs output from the last timestep
 as feature representation of the input (no attention!).
 Remember, you have to account for the zero-padded timesteps!

### Implementation details

The training pipeline (root) is in `train.py`.
The classes for the model definition and dataloading are defined here:
 - `modules/dataloaders.py`
 - `modules/models.py`

but you have to implement the necessary methods.



##### Helpful links
 - http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
 - https://www.sanyamkapoor.com/machine-learning/pytorch-data-loaders/
 - https://github.com/hunkim/PyTorchZeroToAll
 - https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m&disable_polymer=true

