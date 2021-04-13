import torch
from collections import deque
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU = torch.device('cpu')


def predict(model, dataset, criterion):
    preds = deque()
    targets = deque()
    mean_loss = 0
    dataset = DataLoader(dataset)
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            mean_loss += loss.item()
            pred = torch.argmax(pred, dim=1).to(CPU)
            preds.append(pred)
            targets.append(y.to(CPU))
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    print(f'Accuracy: {accuracy:.f5}, f1_score: {f1:.f5}, mean_loss: {mean_loss}')

