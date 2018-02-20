def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False

def count_params(model):
    count = 0
    for param in model.parameters():
        nn = 1
        for s in list(param.size()):
            nn *= s
        count += nn
    return count