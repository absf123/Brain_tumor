# package import
import os
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import CNN
from util import accuracy, plot_acc, plot_loss, set_seed



def train(model, train_loader, optimizer, loss, epoch, num_epochs):
    model.train()
    train_epoch_loss = 0
    train_acc = 0
    start_time = time.time()

    for i, loader in enumerate(train_loader):
        optimizer.zero_grad()
        data, label = loader
        data, label = data.to(device), label.to(device)
        out = model(data)
        train_loss = loss(out, label)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
        train_acc += accuracy(out, label)

    print("[=] EPOCH [{:}/{:}] TIME [{:.3}s]".format(epoch, num_epochs, time.time()-start_time) + \
          " | TRAIN_LOSS [{:.3}] TRAIN_ACC [{:.3}] ".format(
              train_epoch_loss / len(train_loader), train_acc / len(train_loader)))

    return train_epoch_loss, train_acc


def valid(model, valid_loader, loss, epoch, num_epochs):
    valid_epoch_loss = 0
    valid_acc = 0
    start_time = time.time()

    with torch.no_grad():
        model.eval()
        for i, loader in enumerate(valid_loader):
            data, label = loader
            data, label = data.to(device), label.to(device)
            valid_out = model(data)
            valid_loss = loss(valid_out, label)
            valid_epoch_loss += valid_loss.item()
            valid_acc += accuracy(valid_out, label)

    print("[=] EPOCH [{:}/{:}] TIME [{:.3}s]".format(epoch, num_epochs, time.time() - start_time) + \
          " | VAL_LOSS [{:.3}] VAL_ACC [{:.3}] ".format(
              valid_epoch_loss / len(valid_loader), valid_acc / len(valid_loader)))

    return valid_epoch_loss, valid_acc


def trainer(model, train_loader, valid_loader, optimizer, loss, num_epochs):

    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    result = {}
    total_result = []

    for epoch in range(1, num_epochs + 1):
        train_epoch_loss, train_acc = train(model, train_loader, optimizer, loss, epoch, num_epochs)
        valid_epoch_loss, valid_acc = valid(model, valid_loader, loss, epoch, num_epochs)

        train_loss_list.append(train_epoch_loss / len(train_loader))
        train_acc_list.append(train_acc.item() / len(train_loader))
        valid_loss_list.append(valid_epoch_loss / len(valid_loader))
        valid_acc_list.append(valid_acc.item() / len(valid_loader))

    # saving results
    result['train_loss'] = train_loss_list
    result['train_acc'] = train_acc_list
    result['valid_loss'] = valid_loss_list
    result['valid_acc'] = valid_acc_list
    total_result.append(result)

    return total_result

def parser_args():
    # training hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument('--num_epochs', default=50, type=int, help='num_epoch')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='adam betas')

    return parser.parse_args()

if __name__ == "__main__":

    # 0. hyperparameter
    args = parser_args()

    # 1. Data load
    X_train = np.load('Dataset/X_train.npy')
    Y_train = np.load('Dataset/Y_train.npy')
    X_valid = np.load('Dataset/X_valid.npy')
    Y_valid = np.load('Dataset/Y_valid.npy')

    # 2. visualization
    # sns.countplot(Y_train)
    # plt.title("train data label plot")
    # plt.show()
    # sns.countplot(Y_test)
    # plt.title("test data label plot")
    # plt.show()

    # 3. data preparation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed=args.seed)

    X_train = torch.FloatTensor(X_train).to(device)
    Y_train = torch.LongTensor(Y_train).to(device)
    X_valid = torch.FloatTensor(X_valid).to(device)
    Y_valid = torch.LongTensor(Y_valid).to(device)

    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    """ 
    0 : 'glioma_tumor 
    1 : 'meningioma_tumor'
    2 : 'no_tumor'
    3 : 'pituitary_tumor'
    """

    model = CNN(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.wd)
    loss = nn.CrossEntropyLoss()

    # 4. training
    total_result = trainer(model, train_loader, valid_loader, optimizer, loss, args.num_epochs)

    # 5. result & prediction
    plot_acc(total_result)
    plot_loss(total_result)


