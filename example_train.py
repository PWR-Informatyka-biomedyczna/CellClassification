import torch
from torch import nn


from torch.utils.data import DataLoader
from core.dataset import Data, default_transforms, test_transforms
from core.predict import predict

from torchvision.models import resnet18
from core import train, plot_results


def main():
    # 1. setup model
    model = resnet18(True)
    net_fc = model.fc.in_features
    model.fc = torch.nn.Linear(net_fc, 3)
    # 2. Setup train and validate dataset
    data_train = Data(
        path=r'C:\Users\konra\Desktop\data',
        classes=['data_blue_single', 'data_blue_double', 'data_blue_triple'],
        kind='train',
        target_size=(80, 80),
        transforms=default_transforms
    )
    data_val = Data(
        path=r'C:\Users\konra\Desktop\data',
        classes=['data_blue_single', 'data_blue_double', 'data_blue_triple'],
        kind='test',
        target_size=(80, 80),
        transforms=test_transforms
    )
    # 4. Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 5. Train model
    metrics = train(
        model=model,
        max_epochs=20,
        criterion=criterion,
        optimizer=optimizer,
        dataset=data_train,
        verbose=True)
    # 6. Plot metrics
    plot_results(metrics, save=True)
    predict(model, data_val, criterion)


if __name__ == '__main__':
    main()
