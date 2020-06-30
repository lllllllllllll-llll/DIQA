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


class errormapLoss(nn.Module):
    def __init__(self):
        super(errormapLoss, self).__init__()
        pass

    def forward(self, output, gt, r):
        g = torch.sub(output, gt)
        g = torch.mul(g, r)
        l = torch.abs(g)
        l = torch.pow(l, 2)
        loss = torch.mean(l)
        return loss


if __name__ == '__main__':
    parser = ArgumentParser("Pytorch for DIQA")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs_step1", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="LIVE")
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    args = parser.parse_args()

    save_model = "./savemodel/DIQA_step1.pth"

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

    train_dataset = IQADataset(dataset, config, index, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)

    test_dataset = IQADataset(dataset, config, index, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset)

    model = DIQA().to(device)

    criterion1 = errormapLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.9, last_epoch=-1)

    best_SROCC = -1

    print('step1 training ... ')

    # step1
    for epoch in range(args.epochs_step1):
        # train
        status = 'step1'
        model.train()
        LOSS = 0

        for i, ((patches, errormap_gt), label) in enumerate(train_loader):

            patches = patches.to(device)
            errormap_gt = errormap_gt.to(device)
            label = label.to(device)

            # reliability map
            reli_map = []
            alpha = 1
            num = patches.size(0)

            for n in range(num):

                map = 2 / (1 + np.exp(- alpha * np.abs(patches[n]))) - 1
                map = map.numpy() / np.mean(map.numpy())

                map = torch.from_numpy(map)
                map = map.unsqueeze(0)
                map = F.interpolate(map, size=(28, 28))
                map = map.squeeze(0)
                map = map.numpy()
                reli_map.append(map)
            reli_map = torch.Tensor(reli_map).cuda()

            optimizer.zero_grad()
            outputs = model(patches, status)

            loss = criterion1(outputs, errormap_gt, reli_map)

            loss.backward()
            optimizer.step()
            LOSS = LOSS + loss.item()

        train_loss = LOSS / (i + 1)

        print('epoch {}: train loss = {}'.format(epoch + 1, train_loss))

    torch.save(model.state_dict(), save_model)

