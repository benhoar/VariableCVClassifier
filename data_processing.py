import os
import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

length = 1000
folder = {
    'CE': 'data/classification/CE/', 'EC': 'data/classification/EC/', 'E': 'data/classification/E/',
    'ECE': 'data/classification/ECE/', 'DISP': 'data/classification/DISP/'
}

if __name__ == "__main__":
    for j, e in enumerate(folder.keys()):
        dirlist = os.listdir(folder[e])
        dataset = []
        print('Now Handling {} with {} samples'.format(e, len(dirlist)))
        for f in tqdm(dirlist):
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
        with open('data/classification/{}.pkl'.format(e), 'wb') as f:
            pickle.dump(dataset, f)


class Data(Dataset):
    def __init__(self, array, train=False, n_crv=6, noise_mag=1, mask_sr=False):
        # dataformat: 6 * n * 1000
        self.data = array
        self.train = train
        self.n_crv = n_crv
        self.noise_mag = noise_mag
        self.mask_sr = mask_sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         ncrv = np.random.randint(2, 7) if self.train else self.n_crv
        ncrv = self.n_crv
        entry = self.data[idx]['data']
        noise = torch.zeros_like(entry)
        noise[:, (0, 1), :] = torch.normal(noise[:, (0, 1), :], std=self.noise_mag)
        label = self.data[idx]['key']
        mask = torch.zeros_like(entry)
        if ncrv == 1:
            mask[np.random.randint(6)] = 1
        else:
            mask[0] = 1
            mask[5] = 1
            arr = np.random.choice([1, 2, 3, 4], size=(ncrv - 2), replace=False)
            mask[arr] = 1
        if self.mask_sr:
            mask[:, 2, :] = 0
        if self.train:
            return {'data': ((noise + entry) * mask)[torch.randperm(6)], 'label': label}
        else:
            return {'data': ((noise + entry) * mask), 'label': label}


data_files = ['data/classification/EC.pkl', 'data/classification/E.pkl',
              'data/classification/CE.pkl', 'data/classification/ECE.pkl', 'data/classification/DISP.pkl']


def load_data(train_batch_size, test_batch_size, train_size, n_crv, noise_mag, noise_mag_train, mask_sr):
    train, test = [], []
    for file in data_files:
        train_this, test_this = train_test_split(
            pickle.load(open(file, 'rb')), train_size=train_size)
        train += train_this
        test += test_this
    train_data = Data(train, train=True, n_crv=n_crv, noise_mag=0, mask_sr=mask_sr)
    test_data = Data(test, n_crv=n_crv, noise_mag=noise_mag, mask_sr=mask_sr)
    train_loader = DataLoader(
        train_data, batch_size=train_batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)
#     print('Training {}, Testing {}'.format(len(train_data), len(test_data)))
    return train_loader, test_loader, train_data, test_data
