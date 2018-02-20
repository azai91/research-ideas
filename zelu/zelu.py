import torch.nn.functional as F

ZELU_THRESHOLD = 3

def zelu(input, lower=-ZELU_THRESHOLD, upper=ZELU_THRESHOLD):
    right = F.threshold(input - upper, upper, 0)
    left = -F.threshold(lower - input, lower, 0)
    return right + left