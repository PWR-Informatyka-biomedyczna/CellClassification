import torch
from torch import nn


from torch.utils.data import DataLoader
from core.dataset import Data, default_transforms, test_transforms
from core.predict import predict

from torchvision.models import resnet18, vgg11
from core import train, plot_results, DATA


def main():
    # 1. setup model
    model = vgg11(True)
    net_fc = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(net_fc, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 3)  
    )

    data_val = Data(
        path=r'/home/gal/Desktop/data',
        classes=['data_blue_single', 'data_blue_double', 'data_blue_triple'],
        kind='test',
        target_size=(80, 80),
        transforms=test_transforms
    )
    # 4. Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(f'checkpoints/{DATA}/epoch{best_model_epoch + 1}checkpoint.pth'))
    model.eval()
    predict(model, data_val, criterion)


if __name__ == '__main__':
    main()