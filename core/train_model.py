import torch


from core.utils import MetricWriter


def train(model, epochs, criterion, optimizer, train_dataset, val_dataset=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('Start training')

    # setup metric writer
    metric_writer = MetricWriter(len(train_dataset), len(val_dataset))
    for i in range(epochs):
        model.train()
        for x, y in train_dataset:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            metric_writer.add_loss(loss.item(), train=True)
            metric_writer.add_target_pred(pred, y)
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            model.eval()
            for x, y in train_dataset:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                metric_writer.add_loss(loss, train=False)
                metric_writer.add_target_pred(pred, y, train=False)
        metric_writer.calculate_metrics()
