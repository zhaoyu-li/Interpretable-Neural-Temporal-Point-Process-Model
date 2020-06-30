import torch
import random
import numpy as np
import torch.optim as optim

from utils.config import cfg
from utils.parse_args import parse_args
from torch.utils.data import DataLoader
from data.dataset import SyntheticDataset, DemoDataset, collate_fn
from model import INTPP


def train(model, optimizer, scheduler, train_dataloader, device):
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        A = np.zeros((cfg.EVENT_CLASSES, cfg.EVENT_CLASSES))

        for time_seqs, event_seqs in train_dataloader:
            time_seqs = time_seqs.to(device)
            event_seqs = event_seqs.to(device)

            model.zero_grad()

            if (epoch + 1) % cfg.VERBOSE_STEP == 0:
                loss, A_batch = model.forward(time_seqs[:, :cfg.SEQ_LEN - 1], event_seqs[:, :-1], \
                    time_seqs[:, 1:cfg.SEQ_LEN], event_seqs[:, 1:], time_seqs[:, cfg.SEQ_LEN + 1:])
                A += A_batch
            else:
                loss, _ = model.forward(time_seqs[:, :cfg.SEQ_LEN - 1], event_seqs[:, :-1], \
                    time_seqs[:, 1:cfg.SEQ_LEN], event_seqs[:, 1:])

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        scheduler.step()

        A = A / len(train_dataloader)

        print('Epoch {}, epoch loss = {}.'.format(epoch, epoch_loss / len(train_dataloader)))
        
        if (epoch + 1) % cfg.VERBOSE_STEP == 0:
            print('A', A)
            evaluate(model)


def evaluate(model):
    model.eval()
    c, w = model.get_parameters()
    print('c', c)
    print('w', w)


if __name__ == '__main__':
    args = parse_args('Use INTPP to fit and predict on a synthetic dataset.')
    torch.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed(cfg.RAND_SEED)
    torch.cuda.manual_seed_all(cfg.RAND_SEED)
    np.random.seed(cfg.RAND_SEED)
    random.seed(cfg.RAND_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if cfg.Z == 10:
        train_dataset = SyntheticDataset()
    else:
        train_dataset = DemoDataset()

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = INTPP()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_STEP, gamma=cfg.LR_GAMMA)

    train(model, optimizer, scheduler, train_dataloader, device)
