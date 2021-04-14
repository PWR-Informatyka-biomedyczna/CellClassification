import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from core.dataset import Data, default_transforms, test_transforms
from core.predict import predict

from torchvision.models import resnet18, vgg11, googlenet, mobilenet_v2
from core import train, plot_results, DATA
from efficientnet_pytorch import EfficientNet


def take_googlenet():
    model = googlenet(True)
    net_fc = model.fc.in_features
    model.fc = nn.Linear(net_fc, 3)
    return model
    
def take_vgg():
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
    return model

def take_densenet():
    model = densenet121(True)
    net_fc = model.classifier.in_features
    model.classifier = nn.Linear(net_fc, 3)
    return model

def take_efficientnet():
    model = EfficientNet.from_pretrained('efficientnet-b4') 
    net_fc = model._fc.in_features
    model._fc = nn.Linear(net_fc, 3)
    return model

def take_mobilenet():
    model = mobilenet_v2(True)
    net_fc = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(net_fc, 3)
    )
    return model

def main(data_root):
    # 1. setup model
    model = take_efficientnet()
    # 2. Setup train and validate dataset
    data_train = Data(
        path=data_root,
        classes=['data_blue_single', 'data_blue_double', 'data_blue_triple'],
        kind='train',
        target_size=(80, 80),
        transforms=default_transforms
    )
    data_val = Data(
        path=data_root,
        classes=['data_blue_single', 'data_blue_double', 'data_blue_triple'],
        kind='test',
        target_size=(80, 80),
        transforms=test_transforms
    )
    # 4. Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # 5. Train model
    best_model_epoch, metrics = train(
        model=model,
        max_epochs=15,
        criterion=criterion,
        optimizer=optimizer,
        dataset=data_train,
        verbose=True)
    print(best_model_epoch)
    # 6. Plot metrics
    model.load_state_dict(torch.load(f'checkpoints/{DATA}/epoch{best_model_epoch}checkpoint.pth'))
    model.eval()
    plot_results(metrics, save=True)
    predict(model, data_val, criterion)

if __name__ == '__main__':
    main('/home/gal/Desktop/data')
