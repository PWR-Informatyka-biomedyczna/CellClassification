import torch
from torch import nn


from torch.utils.data import DataLoader
from core.dataset import Data, default_transforms, test_transforms


from models.ex_model import CNN
from core import train, plot_results


def main():
    # 1. setup model
    model = CNN(3, 2)
    # 2. Setup train and validate dataset
    data_train = Data(
        path=r'C:\Users\konra\Desktop\cnn_ex',
        classes=['dog', 'akka'],
        kind='train',
        target_size=(128, 128),
        transforms=default_transforms
    )
    data_val = Data(
        path=r'C:\Users\konra\Desktop\cnn_ex',
        classes=['dog', 'akka'],
        kind='test',
        target_size=(128, 128),
        transforms=test_transforms
    )
    # 3. Create dataloader objects
    dtl = DataLoader(data_train, batch_size=1)
    dtv = DataLoader(data_val, batch_size=1)
    # 4. Setup criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 5. Train model
    metrics = train(
        model=model,
        max_epochs=10,
        criterion=criterion,
        optimizer=optimizer,
        train_dataset=dtl,
        val_dataset=dtv,
        verbose=True)
    # 6. Plot metrics
    plot_results(metrics, save=True)


if __name__ == '__main__':
    main()
