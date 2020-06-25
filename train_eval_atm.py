import torch
import random
import numpy as np
import torch.optim as optim

from utils.config import cfg
from utils.parse_args import parse_args
from torch.utils.data import DataLoader
from data.dataset import ATMDataset, collate_fn
from model import RMTPP, INTPP


def train(model, optimizer, scheduler, train_dataloader, test_dataloader, device):
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        scheduler.step()
        epoch_loss = 0
        for time_seqs, event_seqs in train_dataloader:
            time_seqs = time_seqs.to(device)
            event_seqs = event_seqs.to(device)

            model.zero_grad()

            pred_time, pred_event = model.forward(time_seqs[:, :-1], event_seqs[:, :-1])
            loss = model.loss(pred_time, pred_event, time_seqs[:, -1], event_seqs[:, -1])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print('Epoch {}, epoch loss = {}.'.format(epoch, epoch_loss / len(train_dataloader)))

        if (epoch + 1) % 5 == 0:
            evaluate(model, test_dataloader)


def evaluate(model, test_dataloader):
    pred_cnt = np.zeros(cfg.EVENT_CLASSES)
    gt_cnt = np.zeros(cfg.EVENT_CLASSES)
    match_cnt = np.zeros(cfg.EVENT_CLASSES)
    MAE = 0
    cnt = 0

    for time_seqs, event_seqs in test_dataloader:
        time_seqs = time_seqs.to(device)
        event_seqs = event_seqs.to(device)
        pred_times, pred_events = model.predict(time_seqs[:, :-1], event_seqs[:, :-1])
        gt_times, gt_events = time_seqs[:, -1], event_seqs[:, -1]
        cnt += len(pred_times)
        for pred_time, pred_event, gt_time, gt_event in zip(pred_times, pred_events, gt_times, gt_events):
            pred_cnt[pred_event] += 1
            gt_cnt[gt_event] += 1

            if pred_event == gt_event:
                match_cnt[pred_event] += 1

            MAE += abs(pred_time - gt_time)

    # precision = np.zeros(cfg.EVENT_CLASSES)
    # for i in range(cfg.EVENT_CLASSES):
    #     if pred_cnt[i] == 0:
    #         precision[i] = 0
    #     else:
    #         precision[i] = match_cnt[i] / pred_cnt[i]

    precision = match_cnt / pred_cnt
    recall = match_cnt / gt_cnt
    f1_score = 2 * precision * recall / (precision + recall)
    MAE /= cnt

    print('Precision of the pred events is {}, avg is {}'.format(precision, precision.mean()))
    print('Recall of the pred events is {}, avg is {}'.format(recall, recall.mean()))
    print('F1_score of the pred events is {}, avg is {}'.format(f1_score, f1_score.mean()))
    print('MAE of the pred events is', MAE)


if __name__ == '__main__':
    args = parse_args('Use INTPP to fit and predict on a ATM dataset.')
    torch.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed_all(cfg.RAND_SEED)
    np.random.seed(cfg.RAND_SEED)
    random.seed(cfg.RAND_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset = ATMDataset(mode='train')
    test_dataset = ATMDataset(mode='test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = RMTPP()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, optimizer, scheduler, train_dataloader, test_dataloader, device)
