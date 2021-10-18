import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def sensitivity(outputs, labels):
    pass

def specificity(outputs, labels):
    pass


def plot_acc(total_result):
    train_acc = [x['train_acc'] for x in total_result]
    val_acc = [x['valid_acc'] for x in total_result]
    plt.plot(*train_acc)
    plt.plot(*val_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy per epochs')
    plt.show()


def plot_loss(total_result):
    train_loss = [x['train_loss'] for x in total_result]
    val_loss = [x['valid_loss'] for x in total_result]
    plt.plot(*train_loss)
    plt.plot(*val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss per epochs')
    plt.show()
