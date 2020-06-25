import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.config import cfg


class INTPP(nn.Module):
    def __init__(self):
        super(INTPP, self).__init__()
        self.event_classes = cfg.EVENT_CLASSES
        self.event_embed_dim = cfg.EVENT_EMBED_DIM
        self.lstm_hidden_dim = cfg.LSTM_HIDDEN_DIM
        self.hidden_dim = cfg.HIDDEN_DIM
        self.alpha = cfg.ALPHA

        self.event_embed = nn.Embedding(self.event_classes, self.event_embed_dim)
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.lstm = nn.LSTM(self.event_embed_dim + 1, self.lstm_hidden_dim)
        self.time_linear = nn.Linear(self.lstm_hidden_dim, self.event_classes)
        self.event_linear = nn.Linear(self.lstm_hidden_dim + 1, self.event_classes)
        self.c = nn.Parameter(torch.ones(self.event_classes) * 0.8)
        self.w = nn.Parameter(-torch.tensor(0.2))

    def forward(self, input_time, input_event):
        event_embed = self.event_embed(input_event)
        event_embed = self.dropout(event_embed)
        time = input_time.unsqueeze(-1)
        lstm_input = torch.cat([time, event_embed], dim=-1)
        hidden, _ = self.lstm(lstm_input)
        return hidden

    def loss(self, hidden, gt_time, gt_event):
        c = torch.abs(self.c)
        w = self.w

        hp = torch.cat([hidden, gt_time.unsqueeze(-1)], dim=-1)
        hidden = hidden.reshape(-1, self.lstm_hidden_dim)
        lj = self.time_linear(hidden)
        wdt = gt_time.reshape(-1, 1) * w
        lam = torch.exp(lj + wdt) + c
        LAM = (torch.exp(lj + wdt) - torch.exp(lj)) / self.w + torch.matmul(gt_time.reshape(-1, 1), c.unsqueeze(0))

        pred_event = self.event_linear(hp)
        pred_event = pred_event.reshape(-1, self.event_classes)

        event_choice = torch.argmax(pred_event, dim=1).unsqueeze(-1)

        one_hot_choice = torch.zeros(event_choice.shape[0], self.event_classes, device=event_choice.device).\
            scatter_(1, event_choice, 1)

        lamj = one_hot_choice * lam

        lamsum = lamj.sum(1)
        loglam = torch.log(lamsum)

        preLAMlogF = loglam - LAM.sum(1)

        time_loss = -torch.mean(preLAMlogF)

        # time_loss = -(pred_time + self.w * gt_time + self.b +
        #               (torch.exp(pred_time + self.b) - torch.exp(pred_time + self.w * gt_time + self.b)) / self.w).mean()

        # event_loss = F.cross_entropy(pred_event, gt_event)

        # return self.alpha * time_loss + event_loss

        return time_loss

    def predict(self, input_time, input_event):

        return

    def get_parameters(self):
        return torch.abs(self.c).detach(), self.w.detach()



