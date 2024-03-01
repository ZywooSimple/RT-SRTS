import torch

def our_gan(batch):
  xray = [x[0] for x in batch]
  ct = [x[1] for x in batch]
  label=[x[2] for x in batch]
  file_path = [x[3] for x in batch]
  return torch.stack(xray), torch.stack(ct), torch.stack(label),file_path