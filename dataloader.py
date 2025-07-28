import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset, TensorDataset,random_split
from scipy.io import loadmat
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def random_zero_out_pixels(image, zero_fraction=0.1):

    img_np = image.flatten()
    num_pixels = img_np.size
    num_zeros = int(num_pixels * zero_fraction)


    zero_indices = np.random.choice(num_pixels, num_zeros, replace=False)

    mask1 = np.ones(num_pixels)
    mask1[zero_indices] = 0

    mask1=mask1.reshape(image.shape)
    return mask1

def Adultloader(batchSize):

    train_sample = pd.read_csv('./data/train_sample.csv', header=0)
    train_sample = train_sample.values

    # mask
    train_mask_or = random_zero_out_pixels(train_sample, zero_fraction=0.2)
    train_mask_or = torch.from_numpy(train_mask_or)

    mask_expanded = train_mask_or.unsqueeze(dim=1)
    padding = np.ones((27145, 10, 14))
    train_mask = np.concatenate([mask_expanded, padding], axis=1)

    # k近邻的数量
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(train_sample)
    distances, indices = nbrs.kneighbors(train_sample)

    new_train_samples = []
    for i in range(len(train_sample)):
        neighbors = train_sample[indices[i]]
        new_train_samples.append(neighbors)
    new_train_samples = np.array(new_train_samples)

    train_data = new_train_samples* train_mask

    train_data = torch.from_numpy(train_data)

    train_data = train_data.view(-1, 14)
    new_train_samples = torch.from_numpy(new_train_samples)
    new_train_samples = new_train_samples.view(-1, 14)

    numeric_train = train_data[:, :6]
    numeric_train_or=new_train_samples[:, :6]
    categorical_train = train_data[:, 6:]
    categorical_train_or=new_train_samples[:, 6:]

    categories_count = [8, 17, 8, 15, 7, 6, 3, 42]

    # One-Hot 编码
    one_hot_encoded = []
    for i, num_categories in enumerate(categories_count):
        feature = categorical_train[:, i].long()
        one_hot = torch.nn.functional.one_hot(feature, num_classes=num_categories)
        one_hot_encoded.append(one_hot)

    # 拼接所有分类特征的 One-Hot 编码
    one_hot_encoded = torch.cat(one_hot_encoded, dim=1)
    final_train = torch.cat((numeric_train, one_hot_encoded), dim=1)
    final_train=final_train.view(-1,11,112)

    one_hot_encoded_or = []
    for i, num_categories in enumerate(categories_count):
        feature_or = categorical_train_or[:, i].long()
        one_hot_or = torch.nn.functional.one_hot(feature_or, num_classes=num_categories)
        one_hot_encoded_or.append(one_hot_or)

    one_hot_encoded_or = torch.cat(one_hot_encoded_or, dim=1)

    final_train_or = torch.cat((numeric_train_or, one_hot_encoded_or), dim=1)
    final_train_or = final_train_or.view(-1, 11, 112)


    test_sample = pd.read_csv('./data/test_sample.csv', header=0)
    test_sample = test_sample.values

    # mask
    test_mask_or = random_zero_out_pixels(test_sample, zero_fraction=0.5)
    test_mask_or = torch.from_numpy(test_mask_or)
    mask_expanded = test_mask_or.unsqueeze(dim=1)
    padding = np.ones((3017, 10, 14))
    test_mask = np.concatenate([mask_expanded, padding], axis=1)

    new_test_samples = []
    test_nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(train_sample)  # 用训练集来查找测试集的近邻
    test_distances, test_indices = test_nbrs.kneighbors(test_sample)
    for i in range(len(test_sample)):
        test_neighbors = train_sample[test_indices[i]]
        new_test_samples.append(test_neighbors)
    new_test_samples = np.array(new_test_samples)

    test_data = new_test_samples * test_mask
    test_data = torch.from_numpy(test_data)
    test_data = test_data.view(-1, 14)
    new_test_samples = torch.from_numpy(new_test_samples)
    new_test_samples = new_test_samples.view(-1, 14)

    numeric_test = test_data[:, :6]
    categorical_test = test_data[:, 6:]
    numeric_test_or = new_test_samples[:, :6]
    categorical_test_or = new_test_samples[:, 6:]

    categories_count = [8, 17, 8, 15, 7, 6, 3, 42]

    one_hot_encoded = []

    for i, num_categories in enumerate(categories_count):
        feature = categorical_test[:, i].long()
        one_hot = torch.nn.functional.one_hot(feature, num_classes=num_categories)
        one_hot_encoded.append(one_hot)

    one_hot_encoded = torch.cat(one_hot_encoded, dim=1)
    final_test = torch.cat((numeric_test, one_hot_encoded), dim=1)
    final_test = final_test.view(-1, 11, 112)

    one_hot_encoded_or = []
    for i, num_categories in enumerate(categories_count):
        feature_or = categorical_test_or[:, i].long()
        one_hot_or = torch.nn.functional.one_hot(feature_or, num_classes=num_categories)
        one_hot_encoded_or.append(one_hot_or)


    one_hot_encoded_or = torch.cat(one_hot_encoded_or, dim=1)
    final_test_or = torch.cat((numeric_test_or, one_hot_encoded_or), dim=1)
    final_test_or = final_test_or.view(-1, 11, 112)
    train_dataset = list(zip(train_sample,train_mask_or, final_train,final_train_or))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True
    )


    test_dataset = list(zip(test_sample, test_mask_or,final_test,final_test_or))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True
    )
    return train_loader, test_loader
