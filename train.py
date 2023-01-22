from data_processing import load_data, folder
from network import Resnet50
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import numpy as np
import torch
from tqdm import tqdm

for i in range(8):
    train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=3500, 
            n_crv=6, noise_mag=0, noise_mag_train=0.3, mask_sr=False)
    model = Resnet50()
    model.fit(train, epoch=100)
    target, pred = model.test(test, epoch=1)
    torch.save(model.model.state_dict(), 'resnet18-3-{}.pth'.format(i))
    print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))

    train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=3500, 
            n_crv=6, noise_mag=0, noise_mag_train=0, mask_sr=False)
    model = Resnet50()
    model.fit(train, epoch=100)
    target, pred = model.test(test, epoch=1)
    torch.save(model.model.state_dict(), 'resnet18-0-{}.pth'.format(i))
    print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))

    train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=4096, train_size=3500, 
            n_crv=6, noise_mag=0, noise_mag_train=0.1, mask_sr=False)
    model = Resnet50()
    model.fit(train, epoch=100)
    target, pred = model.test(test, epoch=1)
    torch.save(model.model.state_dict(), 'resnet18-1-{}.pth'.format(i))
    print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))
