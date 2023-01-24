from resnet import resnet50, resnet18
import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

class Resnet50:
    def __init__(self, device='cuda') -> None:
        self.device = device
        self.model = resnet18(num_classes=3).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),weight_decay=1e-5)
    
    def fit(self, train_loader, epoch=100):
        self.model.train()
        loss_list = []
        for _ in tqdm(range(epoch)):
            train_loss = 0
            for batch, data in enumerate(train_loader):
                X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                classes = torch.argmax(pred, dim=1)
                train_loss += (classes != y).float().sum()
            loss_list.append(train_loss)
        return loss_list
    
    def test(self, test_loader, epoch=1):
        self.model.eval()
        self.target = []
        self.prediction = []
        with torch.no_grad():
            for _ in tqdm(range(epoch)):
                for batch, data in enumerate(test_loader):
                    X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                    _, pred = torch.max(self.model(X).data, 1)
                    self.target += y.tolist()
                    self.prediction += pred.detach().tolist()
#         self.cm = confusion_matrix(self.target, self.prediction, normalize='true')
#         self.acc = accuracy_score(self.target, self.prediction)
        return self.target, self.prediction
    
class ResnetReg:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = resnet18(num_classes=1).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

    def fit(self, train_loader, epoch=100):
        self.model.train()
        loss_list = []
        for _ in tqdm(range(epoch)):
            train_loss = 0
            for batch, data in enumerate(train_loader):
                X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                y[y >= 0.5] = 1
                y[y < 0.5] = 0
                pred = self.model(X).view(-1)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            loss_list.append(train_loss)
        return loss_list

    def test(self, test_loader, epoch=1):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for _ in tqdm(range(epoch)):
                for batch, data in enumerate(test_loader):
                    X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                    pred = self.model(X).view(-1)
                    loss += self.loss_fn(pred, y).item()
        print(loss / epoch)
        return pred.cpu().detach(), y.cpu().detach()

