import sys

import torch
import torch.nn as nn
import Preprocess


train_iter = build_iterator("data/en/train.txt", batch_size=32, device='cpu')
dev_iter = build_iterator("data/en/testa.txt", batch_size=32, device='cpu')
test_iter = build_iterator("data/en/testb.txt", batch_size=32, device='cpu')