import random, os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from datasets import ArrhythmiaDataset, SleepEdfDataset, MotorImageryDataset

def log_selector(selector1, selector2, mask):
    assert selector1.shape[1] == selector2.shape[1], "different selector heads"

    s1 = selector1[0].long().cpu().detach()
    s2 = selector2[0].cpu().detach()
    mask = mask[0].cpu().detach()

    indices = []
    j = 0
    while len(indices) < 50 and j < s1.shape[1]:
        # Iterating over the mask like this may only work for single channel data?
        if mask[0][j]:
            indices.append(j)
        j += np.random.randint(10)

    print("selectors")
    for i in range(len(s2)):
        print(s2[i].item(), end="\t")
        print(s1[i].sum().item(), "selected", end="\t")
        for j in indices:
            print(s1[i, j].item(), end="")
        print("")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data(name="sleep_edf", base_dir=".", batch_size=64, val_subject_ids=[0,1], test_subject_ids=[2,7,12]):
    if name=='arrhythmia':
        dataset = ArrhythmiaDataset(base_dir+'/'+'datasets/ecg', normalize=False)
    elif name=='sleep_edf':
        dataset = SleepEdfDataset(base_dir+'/'+'sleep_edf/prepared', 'fpz_cz', val_subject_ids, test_subject_ids)
    elif name=='motor_imagery':
        dataset = MotorImageryDataset(base_dir+'/'+'motor_imagery/prepared', val_subject_ids, test_subject_ids)
    else:
        raise ValueError("Value for 'name' must be 'arrhythmia' or 'sleep_edf' or 'motor_imagery'")

    train_dl = DataLoader(
        TensorDataset(dataset.X_train, dataset.y_train), batch_size, shuffle=True, num_workers=4, persistent_workers=True
    )
    val_dl = DataLoader(
        TensorDataset(dataset.X_val, dataset.y_val), batch_size, shuffle=False, num_workers=4, persistent_workers=True
    )
    test_dl = DataLoader(
        TensorDataset(dataset.X_test, dataset.y_test), batch_size, shuffle=False, num_workers=4, persistent_workers=True
    )

    max_len = dataset.sequence_length
    return train_dl, val_dl, test_dl, dataset.class_weights, max_len

def my_optim(params, lr, wd, warmup_steps, use_scheduler=False):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad == True, params),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    # optimizer = torch.optim.RMSprop(
    #     filter(lambda p: p.requires_grad == True, params),
    #     lr=lr,
    #     weight_decay=wd
    # )

    if not use_scheduler:
        return optimizer

    def lr_foo(epoch):
        if epoch < warmup_steps:
            # warm up lr
            lr_scale = 0.1 ** (warmup_steps - epoch)
        else:
            lr_scale = 0.95 ** epoch
        return lr_scale

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)

    return [optimizer], [scheduler]
