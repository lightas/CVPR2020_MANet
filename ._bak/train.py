import torch
import os
from config import cfg
from networks.deeplabv3plus import deeplabv3plus

model = deeplabv3plus(cfg)
print(model)

