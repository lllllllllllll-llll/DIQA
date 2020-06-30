import torch
import os
import numpy as np
from scipy import stats
import yaml
from argparse import ArgumentParser
import random
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import h5py

from network import DIQA
from IQADataset import IQADataset

def get_indexNum(config, index, status):

    test_ratio = config['test_ratio']
    train_ratio = config['train_ratio']
    trainindex = index[:int(train_ratio * len(index))]
    testindex = index[int((1 - test_ratio) * len(index)):]
    train_index, test_index = [], []

    ref_ids = []
    for line0 in open("./data/live/ref_ids.txt", "r"):
        line0 = float(line0[:-1])
        ref_ids.append(line0)
    ref_ids = np.array(ref_ids)

    for i in range(len(ref_ids)):
        if (ref_ids[i] in trainindex):
            train_index.append(i)
        elif (ref_ids[i] in testindex):
            test_index.append(i)

    if status == 'train':
        index = train_index
    if status == 'test':
        index = test_index

    return len(index)



if __name__ == '__main__':
    parser = ArgumentParser("Pytorch for DIQA")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs_step2", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dataset", type=str, default="LIVE")
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    args = parser.parse_args()

    model_path = "./savemodel/DIQA_step1.pth"
    save_model = "./savemodel/DIQAmodel.pth"

    seed = random.randint(10000000, 99999999)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("seed:", seed)

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

    index = []
    if args.dataset == "LIVE":
        print("dataset: LIVE")
        index = list(range(1, 30))
        random.shuffle(index)
    elif args.dataset == "TID2013":
        print("dataset: TID2013")
        index = list(range(1, 26))

    print('rando index', index)

    dataset = args.dataset
    testnum = get_indexNum(config, index, "test")
    print('testnum', testnum)

    train_dataset = IQADataset(dataset, config, index, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)

    test_dataset = IQADataset(dataset, config, index, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset)

    model = DIQA().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.9, last_epoch=-1)

    best_SROCC = -1

    model.load_state_dict(torch.load(model_path))
    print('step2 training ... ')
    status = 'step2'
    # step2
    for epoch in range(args.epochs_step2):
        # train
        model.train()
        LOSS = 0
        for i, ((patches, errormap_gt), label) in enumerate(train_loader):
            patches = patches.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(patches, status)

            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            LOSS = LOSS + loss.item()
        train_loss = LOSS / (i + 1)

        # test
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L = 0
        with torch.no_grad():
            for i, ((patches, errormap_gt), label) in enumerate(test_loader):
                y_test[i] = label.item()

                patches = patches.to(device)
                label = label.to(device)

                outputs = model(patches, status)

                score = outputs.mean()
                y_pred[i] = score
                loss = criterion(score, label[0])
                L = L + loss.item()
        test_loss = L / (i + 1)

        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

        print("Epoch {} Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC,
                                                                                                            PLCC,
                                                                                                            KROCC,
                                                                                                            RMSE))

        if SROCC > best_SROCC:
            print("Update Epoch {} best valid SROCC".format(epoch))
            print("Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                       SROCC,
                                                                                                       PLCC,
                                                                                                       KROCC,
                                                                                                       RMSE))
            torch.save(model.state_dict(), save_model)
            best_SROCC = SROCC

    # final test
    model.load_state_dict(torch.load(save_model))
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L = 0
        for i, ((patches, errormap_gt), label) in enumerate(test_loader):
            y_test[i] = label.item()
            patches = patches.to(device)
            label = label.to(device)

            outputs = model(patches, status)

            score = outputs.mean()
            y_pred[i] = score
            loss = criterion(score, label[0])
            L = L + loss.item()
    test_loss = L / (i + 1)

    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
    print("Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                     SROCC,
                                                                                                     PLCC,
                                                                                                     KROCC,
                                                                                                     RMSE))

