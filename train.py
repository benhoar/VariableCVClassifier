from data_processing import load_data, folder
from network import Resnet50
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import numpy as np
import torch
from tqdm import tqdm

# train_size reduced from 3500 to 3000
# test_size reduced from 4096 to 3500
# epochs changed from from 100 to 10
# mask_sr deleted from load_data for all


if __name__ == "__main__":
    lcl_device='cuda'
    data_type = 'mixed'
    for i in range(8):
        # 0.1 training noise
        training_noise = 0.1
        train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=3000, train_size=2500, 
            n_crv=6, noise_mag=0, noise_mag_train=training_noise)
        model = Resnet50(device=lcl_device)
        model.fit(train, epoch=40) 
        target, pred = model.test(test, epoch=1)
        torch.save(model.model.state_dict(), 'resnet18-{}-{}-{}.pth'.format(i, training_noise, data_type))
        print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))

        # 0.01 training noise
        training_noise = 0.01
        train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=3000, train_size=2500, 
            n_crv=6, noise_mag=0, noise_mag_train=training_noise)
        model = Resnet50(device=lcl_device)
        model.fit(train, epoch=40) 
        target, pred = model.test(test, epoch=1)
        torch.save(model.model.state_dict(), 'resnet18-{}-{}-{}.pth'.format(i, training_noise, data_type))
        print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))

        # 0 training noise
        training_noise = 0
        train, test, _, _ = load_data(
            train_batch_size=256, test_batch_size=3000, train_size=2500, 
            n_crv=6, noise_mag=0, noise_mag_train=training_noise)
        model = Resnet50(device=lcl_device)
        model.fit(train, epoch=40) 
        target, pred = model.test(test, epoch=1)
        torch.save(model.model.state_dict(), 'resnet18-{}-{}-{}.pth'.format(i, training_noise, data_type))
        print('Accuracy: {:.6f}'.format(accuracy_score(target, pred)))
