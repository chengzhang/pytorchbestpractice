import torch


# check GPU available
t = torch.rand(3, 4)
if torch.cuda.is_available():
    t = t.to('cuda')

