import torch
import os
import os.path as osp
import csv
import numpy as np
import math
import pickle

from collections import defaultdict
from utils.config import cfg
from torch.utils.data import Dataset


class ATMDataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.dataset_name = 'atm_train.csv'
        else:
            self.dataset_name = 'atm_test.csv'

        self.seq_len = cfg.SEQ_LEN

        self.time_preprocess = []
        self.event_preprocess = []
        self.atm_name = defaultdict(int)
        self.atm_idx = 1

        dataset_reader = csv.reader(open(osp.abspath(osp.join(cfg.DATA_DIR, self.dataset_name))))
        next(dataset_reader)

        for data in dataset_reader:
            if self.atm_name[data[0]] == 0:
                self.atm_name[data[0]] = self.atm_idx
                self.time_preprocess.append([])
                self.event_preprocess.append([])
                self.atm_idx += 1

            self.time_preprocess[self.atm_name[data[0]] - 1].append(float(data[1]))
            self.event_preprocess[self.atm_name[data[0]] - 1].append(int(data[2]))

        self.time_dataset = []
        self.event_dataset = []

        for time_seq, event_seq in zip(self.time_preprocess, self.event_preprocess):
            for i in range(len(time_seq) - self.seq_len + 1):
                if time_seq[i + self.seq_len - 1] - time_seq[i] > 1:
                    continue
                self.time_dataset.append(time_seq[i: i + self.seq_len])
                self.event_dataset.append(event_seq[i: i + self.seq_len])

    def __len__(self):
        return len(self.time_dataset)

    def __getitem__(self, idx):
        return self.time_dataset[idx], self.event_dataset[idx]


class DemoDataset(Dataset):
    def __init__(self):
        self.seq_len = cfg.SEQ_LEN

        self.time_dataset = []
        self.event_dataset = []

        self.dataset_name = '2DHawkes.txt'

        f = open(osp.abspath(osp.join(cfg.DATA_DIR, self.dataset_name)))

        for line in f.readlines():
            line = line.strip().split(',')
            time_seq = [float(t.split()[0]) for t in line]
            event_seq = [int(t.split()[1]) for t in line]
            if len(line) < self.seq_len:
                continue
            for i in range(len(line) - self.seq_len + 1):
                self.time_dataset.append(time_seq[i: i + self.seq_len])
                self.event_dataset.append(event_seq[i: i + self.seq_len])

    def __len__(self):
        return len(self.time_dataset)

    def __getitem__(self, idx):
        return self.time_dataset[idx], self.event_dataset[idx]


class SyntheticDataset(Dataset):
    def __init__(self):
        self.time_dataset_name = 'synthetic_time_train.pkl'
        self.event_dataset_name = 'synthetic_event_train.pkl'

        self.seq_len = cfg.GEN_MAX_SEQ_LEN

        if not os.path.exists(osp.abspath(osp.join(cfg.DATA_DIR, self.time_dataset_name))):
            self.generate_data()

        self.time_sequences = pickle.load(open(osp.abspath(osp.join(cfg.DATA_DIR, self.time_dataset_name)), 'rb'))
        self.event_sequences = pickle.load(open(osp.abspath(osp.join(cfg.DATA_DIR, self.event_dataset_name)), 'rb'))

        self.time_dataset = []
        self.event_dataset = []

        for i in range(0, len(self.time_sequences), self.seq_len):
            self.time_dataset.append(self.time_sequences[i: i+self.seq_len])
            self.event_dataset.append(self.event_sequences[i: i+self.seq_len])

    def generate_data(self):
        def cond_intensity(t, events, dim):
            exp_decay = [[math.exp(-cfg.W * (t - t_i)) for t_i in d_events] for d_events in events]
            return cfg.MU[dim] + np.sum(
                [cfg.A[d][dim] * exp_decay[d][i] for d in range(cfg.Z) for i in range(len(events[d]))])

        def generate():
            events = [[] for _ in range(cfg.Z)]
            intensities = [[cfg.MU[d]] for d in range(cfg.Z)]
            gen_time = []
            gen_event = []
            s = 0
            cnt = 0
            while cnt < cfg.GEN_MAX_SEQ_LEN:
                lamb_bar = np.sum([cond_intensity(s, events, d) for d in range(cfg.Z)])
                u = np.random.rand()
                w = - math.log(u) / lamb_bar
                s = s + w
                D = np.random.rand()
                lamb = [cond_intensity(s, events, d) for d in range(cfg.Z)]
                if D * lamb_bar <= np.sum(lamb):
                    d = 0
                    while d < cfg.Z:
                        if D * lamb_bar <= np.sum(lamb[:d + 1]):
                            break
                        d += 1
                    events[d].append(s)
                    intensities[d].append(lamb[d])
                    gen_time.append(s)
                    gen_event.append(d)
                    cnt += 1
            return gen_time, gen_event

        self.time_dataset = []
        self.event_dataset = []

        for i in range(cfg.GEN_TRAIN_SEQ_NUM):
            print(i)
            gen_time, gen_event = generate()
            self.time_dataset.extend(gen_time)
            self.event_dataset.extend(gen_event)

        pickle.dump(self.time_dataset, open(osp.abspath(osp.join(cfg.DATA_DIR, 'synthetic_time_train.pkl')), 'wb'))
        pickle.dump(self.event_dataset, open(osp.abspath(osp.join(cfg.DATA_DIR, 'synthetic_event_train.pkl')), 'wb'))

    def __len__(self):
        return len(self.time_dataset)

    def __getitem__(self, idx):
        return self.time_dataset[idx], self.event_dataset[idx]


def collate_fn(batch_data):
    time_seqs = []
    event_seqs = []
    for time_seq, event_seq in batch_data:
        time_seq = np.array([time_seq[0]] + time_seq)
        time_seq = np.diff(time_seq)
        time_seqs.append(time_seq)
        event_seqs.append(event_seq)
    return torch.FloatTensor(time_seqs), torch.LongTensor(event_seqs)


if __name__ == '__main__':
    # atm = ATMDataset(mode='train')
    # synthetic = SyntheticDataset()
    synthetic = DemoDataset()
    print(len(synthetic[0][0]))
    # print(atm[0])
    # for data in atm:
    #     print(len(data))



