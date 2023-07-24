import torch
def get_weight(dataset_raw):
    train_labels = dataset_raw.train_lables
    num_classes = dataset_raw.num_classes

    if num_classes == 2:
        num_pos = (train_labels == 1).sum().item()
        num_neg = (train_labels == 0).sum().item()
        weight = num_neg / num_pos
    else:
        weight = torch.nn.functional.one_hot(train_labels, num_classes=num_classes).sum(0).float()
        weight  = weight.sum()/weight
        weight  = weight/ weight.sum()

        # weight = torch.nn.functional.softmax(-weight.float()) * len(weight)
        weight = weight.flatten().tolist()
    return weight