import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.config import cfg


class INTPP(nn.Module):
    def __init__(self):
        super(INTPP, self).__init__()
        self.event_classes = cfg.EVENT_CLASSES
        self.event_embed_dim = cfg.EVENT_EMBED_DIM
        self.lstm_hidden_dim = cfg.LSTM_HIDDEN_DIM
        self.alpha = cfg.ALPHA

        self.event_embed = nn.Embedding(self.event_classes, self.event_embed_dim)
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.lstm = nn.LSTM(self.event_embed_dim + 1, self.lstm_hidden_dim, batch_first=True)
        self.time_linear = nn.Linear(self.lstm_hidden_dim, self.event_classes)
        self.event_linear = nn.Linear(self.lstm_hidden_dim + 1, self.event_classes)
        self.c = nn.Parameter(torch.ones(self.event_classes) * 0.8)
        self.w = nn.Parameter(-torch.tensor(0.2))

    def forward(self, input_time, input_event, gt_time, gt_event, base_time=None):
        event_embed = self.event_embed(input_event)
        event_embed = self.dropout(event_embed)
        time = input_time.unsqueeze(-1)
        lstm_input = torch.cat([time, event_embed], dim=-1)
        hidden, _ = self.lstm(lstm_input)

        c = torch.abs(self.c)
        w = self.w

        h = torch.cat([hidden, gt_time.unsqueeze(-1)], dim=-1)
        h = h.reshape(-1, self.lstm_hidden_dim + 1)

        hidden = hidden.reshape(-1, self.lstm_hidden_dim)
        lj = self.time_linear(hidden)
        
        wdt = gt_time.reshape(-1, 1) * w
        lam = torch.exp(lj + wdt) + c
        LAM = (torch.exp(lj + wdt) - torch.exp(lj)) / w + torch.matmul(gt_time.reshape(-1, 1), c.unsqueeze(0))

        pred_event = self.event_linear(h)

        event_choice = gt_event.reshape(-1, 1)

        one_hot_choice = torch.zeros(event_choice.shape[0], self.event_classes, device=event_choice.device). \
            scatter_(1, event_choice, 1)

        lamj = one_hot_choice * lam

        lamsum = lamj.sum(1)
        loglam = torch.log(lamsum)

        preLAMlogF = loglam - LAM.sum(1)

        time_loss = -torch.mean(preLAMlogF)

        if cfg.USE_EVENT_LOSS:
            event_loss = F.cross_entropy(pred_event, gt_event.reshape(-1))
        else:
            event_loss = 0

        A = None

        if base_time is not None:
            base_time = base_time.cpu()
            gt_event = gt_event.clone().cpu()
            w = w.clone().detach().cpu()
            lj = lj.clone().detach().cpu().view(-1, cfg.SEQ_LEN - 1, cfg.EVENT_CLASSES)
            A_list = []
            for j in range(cfg.SEQ_LEN - 1):
                a = []
                for d in range(cfg.EVENT_CLASSES):
                    b = torch.exp(lj[:, j, d]).unsqueeze(-1).cpu().numpy()
                    mark = []
                    tmp_A = []
                    for dd in range(cfg.EVENT_CLASSES):
                        mark.append((gt_event == dd).float())
                        tmp = (torch.exp(w * (base_time[:, j, None] - base_time[:, :j + 1])) * mark[dd][:, :j + 1])
                        tmp = tmp.sum(dim=-1).unsqueeze(-1)
                        tmp_A.append(tmp.cpu().numpy())
                    A_ = np.concatenate(tmp_A, axis=1)
                    clf = LinearRegression()
                    clf.fit(A_, b)
                    a.append(clf.coef_)
                a = np.concatenate(a, axis=0)
                A_list.append(a)

            A = np.mean(A_list, axis=0)

        return (self.alpha * time_loss + event_loss), A

    def predict(self, input_time, input_event):
        event_embed = self.event_embed(input_event)
        event_embed = self.dropout(event_embed)
        time = input_time.unsqueeze(-1)
        lstm_input = torch.cat([time, event_embed], dim=-1)
        hidden, _ = self.lstm(lstm_input)

        c = torch.abs(self.c)
        w = self.w

        hidden = hidden.reshape(-1, self.lstm_hidden_dim)
        lj = self.time_linear(hidden)

        pred_time = torch.tensor([
            self.newton_update(0.5, 0.01, lj[idx].sum().detach(), w.detach(), c.sum().detach())
            for idx in range(lj.shape[0])
        ]).to(input_time.device)

        h = torch.cat([hidden, pred_time.unsqueeze(-1)], dim=-1)

        pred_event = self.event_linear(h.reshape(-1, self.lstm_hidden_dim + 1))
        pred_event = torch.argmax(pred_event, dim=-1)

        return pred_time.detach(), pred_event.detach()

    def newton_update(self, u, dt0, lj_, w, c):
        def K(d):
            return (torch.exp(lj_ + w * d) - torch.exp(lj_)) / w + c * d + torch.log(1 - u)

        def K_p(d):
            return torch.exp(lj_ + w * d) + c

        u = torch.tensor(u)
        dt0 = torch.tensor(dt0)
        dt = dt0
        dt1 = dt - K(dt) / K_p(dt)

        cnt = 0

        while torch.abs(dt1 - dt) > 0.01 and cnt < 100:
            dt = dt1
            dt1 = dt - K(dt) / K_p(dt)
            cnt += 1

        return dt1

    def get_parameters(self):
        return torch.abs(self.c).detach(), self.w.detach()
