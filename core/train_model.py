import torch


from core.utils import MetricWriter, EarlyStopper, log_msg


def train(model, max_epochs, criterion, optimizer, train_dataset, val_dataset, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    log_msg(f'Device: {device}', verbose)
    log_msg(f'Train samples: {len(train_dataset)}, validate samples: {len(val_dataset)}', verbose)
    log_msg(f'Start training, loss: {criterion}, optimizer: {optimizer}', verbose)
    # setup metric writer
    metric_writer = MetricWriter(len(train_dataset), len(val_dataset))
    early_stopping = EarlyStopper(patience=7)
    for i in range(max_epochs):
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
        early_stopping.stop(metric_writer.last_val_loss)
        log_msg(f'Epoch {i + 1} / {max_epochs}, {metric_writer.get_last_epoch_info()}', verbose)
        torch.save(model.state_dict(), f'checkpoints/epoch{i + 1}checkpoint.pth')
        log_msg(f'Saved model to checkpoints/epoch{i + 1}checkpoint.pth')
        if early_stopping:
            log_msg('Early stopping', verbose)
            break
    return metric_writer.metrics

