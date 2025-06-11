import torch
print(torch.version.cuda)          # still 11.8
print(torch.backends.cudnn.version())  # ≥ 93000