import torch

def setup_torch_seed(seed=1):
    # pytorchに関連する乱数シードの固定を行う．
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
