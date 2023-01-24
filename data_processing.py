import os
import shutil
import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

length = 1000
folder = { 
           'EC1':'D:/CV_Project_2/9ModelData/EC1/lin_srs/',
           'E':'D:/CV_Project_2/9ModelData/E/lin_srs/',
           'ECP': 'D:/CV_Project_2/9ModelData/ECP/lin_srs/'
         }


if __name__ == "__main__":
    for j, e in enumerate(folder.keys()):
        dirlist = os.listdir(folder[e])
        dirlist = [file for file in dirlist if file.endswith('txt')]
        dataset = []
        print('Now Handling {} with {} samples'.format(e, len(dirlist)))
        for f in tqdm(dirlist):
            try: 
                read = np.genfromtxt(folder[e] + f, skip_header=1, delimiter=',')
                scan_rate, scan_rate_idx = np.unique(read[:, 2], return_index=True)
                mat_val = cv2.resize(
                    np.vstack([x[:, 1] for x in np.split(read, scan_rate_idx[1:])]), (length, 6))
                mat_scan = np.tile(scan_rate, (length // 2, 1)).T
                mat_val1 = mat_val[:, :length // 2]
                mat_val2 = mat_val[:, :-(length // 2 + 1):-1]
                mat = np.stack((mat_val1, mat_val2, mat_scan), axis=1)
                dataset.append({'data': torch.tensor(
                    mat, dtype=torch.float32), 'label': e, 'file': f, 'key': j})
            except:
                shutil.move(folder[e] + f, f'D:/CVClassify/errorfolder/{e}/{f}')
        with open('D:/CVClassify/classification/{}.pkl'.format(e), 'wb') as f:
            pickle.dump(dataset, f)


# deleted mask_sr variable of Data class because data is now separated into 
# folders and does not need to be evaluated at run time

# Data inherits pytorch.utils.Dataset, which is a class that simplifies
# The storing of data and their labels

class Data(Dataset):
    def __init__(self, array, train=False, n_crv=6, noise_mag=1):
        # dataformat: 6 * n * 1000
        self.data = array
        self.train = train
        self.n_crv = n_crv
        self.noise_mag = noise_mag

    def __len__(self):
        return len(self.data)

    def __visualize(self, data):
        print(data.size())
        colors = ['red', 'orange', 'green', 'blue', 'indigo', 'violet']
        for curve, color in zip(data, colors):
            plt.scatter(range(len(curve[0])), curve[0], color=color)
            plt.scatter(range(len(curve[1])), curve[1], color=color)
        plt.show()
    
    # major updates: get item simplified to, there is no masking of random scan rate values
    # so masking was REMOVED, also permutation of training data was REMOVED, I don't see why
    # this was being down, the order of the curves is relevant information
    
    def __getitem__(self, idx):
        entry = self.data[idx]['data']
        noise = torch.zeros_like(entry)
        noise[:, (0, 1), :] = torch.normal(noise[:, (0, 1), :], std=self.noise_mag)
        label = self.data[idx]['key']
        return {'data': ((noise + entry)), 'label': label}


data_files = ['D:/CVClassify/classification/ECP.pkl', 'D:/CVClassify/classification/E.pkl', 'D:/CVClassify/classification/EC1.pkl']


def load_data(train_batch_size, test_batch_size, train_size, n_crv, noise_mag, noise_mag_train):
    train, test = [], []
    for file in data_files:
        train_this, test_this = train_test_split(
            pickle.load(open(file, 'rb')), train_size=train_size)
        train += train_this
        test += test_this
    train_data = Data(train, train=True, n_crv=n_crv, noise_mag=noise_mag_train) #noise mag train was hard coded to 0
    test_data = Data(test, n_crv=n_crv, noise_mag=noise_mag)
    train_loader = DataLoader(
        train_data, batch_size=train_batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    #print('Training {}, Testing {}'.format(len(train_data), len(test_data)))
    return train_loader, test_loader, train_data, test_data
