import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler


from core.utils import MetricWriter, MetricWriterKFold, EarlyStopper, log_msg, DATA


def train(model, max_epochs, criterion, optimizer, dataset, k_folds=5, batch_size=4, patience=7, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    log_msg(f'Device: {device}', verbose)
    log_msg(f'Dataset size: {len(dataset)}', verbose)
    log_msg(f'Start training, loss: {criterion}, optimizer: {optimizer}', verbose)
    # setup metric writer
    splitter = KFold(k_folds, shuffle=True)
    metric_writer = MetricWriter()
    early_stopping = EarlyStopper(patience=patience)
    for i in range(max_epochs):
        best_idx = i + 1
        for j, (train_dataset, val_dataset) in enumerate(splitter.split(dataset)):
            train_sampler = SubsetRandomSampler(train_dataset)
            val_sampler = SubsetRandomSampler(val_dataset)

            train_dataset = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_dataset = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
            kfold_writer = MetricWriterKFold(len(train_dataset), len(val_dataset))
            model.train()
            for x, y in train_dataset:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                kfold_writer.add_loss(loss.item(), train=True)
                kfold_writer.add_target_pred(pred, y)
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                model.eval()
                for x, y in val_dataset:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    kfold_writer.add_loss(loss.item(), train=False)
                    kfold_writer.add_target_pred(pred, y, train=False)
            kfold_writer.calculate_metrics_fold()
            log_msg(f'Epoch {i + 1}/{max_epochs}, fold {j + 1}/{k_folds} => {kfold_writer.get_last_epoch_info()}', verbose)
        metric_writer.add_metrics(kfold_writer.metrics)
        early_stopping.stop(metric_writer.last_val_loss)
        log_msg(f'Epoch {i + 1} / {max_epochs}, {metric_writer.get_last_epoch_info()}', verbose)
        torch.save(model.state_dict(), f'checkpoints/{DATA}/epoch{i + 1}checkpoint.pth')
        log_msg(f'Saved model to checkpoints/epoch{i + 1}checkpoint.pth')
        if early_stopping:
            best_idx -= patience
            log_msg('Early stopping', verbose)
            break
    return best_idx, metric_writer.metrics

