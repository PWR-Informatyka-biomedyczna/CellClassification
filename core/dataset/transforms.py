from torchvision import transforms


def default_transforms(target_size):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])


def test_transforms(target_size):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
