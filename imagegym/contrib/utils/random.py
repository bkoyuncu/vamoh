import torch


def get_permutation(original_len, device='auto', seed=0):
    if device == 'auto':
        generator = torch.Generator(device='cpu')
    else:
        generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    perm = torch.randperm(original_len, generator=generator)
    return perm


def num_classes_fn(data) -> int:
    r"""Returns the number of classes in the dataset."""
    y = data.y
    if y is None:
        return 0
    elif y.numel() == y.size(0) and not torch.is_floating_point(y):
        return int(data.y.max()) + 1
    elif y.numel() == y.size(0) and torch.is_floating_point(y):
        return torch.unique(y).numel()
    else:
        return data.y.size(-1)


def normal(loc, scale, shape, device='auto', seed=0):
    if device == 'auto':
        generator = torch.Generator(device='cpu')
    else:
        generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    normal = torch.distributions.Normal(loc=loc, scale=scale)
    shape = normal._extended_shape(shape)
    with torch.no_grad():
        return torch.normal(normal.loc.expand(shape), normal.scale.expand(shape), generator=generator)


def bernoulli(probs, device='auto', seed=0):
    if device == 'auto':
        generator = torch.Generator(device='cpu')
    else:
        generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.bernoulli(probs, generator=generator).to(torch.bool)
