import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.config import cfg
from scipy.integrate import quad


class RMTPP(nn.Module):
    def __init__(self):
        super(RMTPP, self).__init__()
        self.event_classes = cfg.EVENT_CLASSES
        self.event_embed_dim = cfg.EVENT_EMBED_DIM
        self.lstm_hidden_dim = cfg.LSTM_HIDDEN_DIM
        self.hidden_dim = cfg.HIDDEN_DIM
        self.alpha = cfg.ALPHA

        self.event_embed = nn.Embedding(self.event_classes, self.event_embed_dim)
        self.lstm = nn.LSTM(self.event_embed_dim + 1, self.lstm_hidden_dim)
        self.hidden_embed = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.time_linear = nn.Linear(self.hidden_dim, 1)
        self.event_linear = nn.Linear(self.hidden_dim, self.event_classes)
        self.w = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_time, input_event):
        event_embed = self.event_embed(input_event)
        time = input_time.unsqueeze(-1)
        lstm_input = torch.cat([time, event_embed], dim=-1)
        hidden, _ = self.lstm(lstm_input)
        hidden = hidden[:, -1, :]
        hidden_embed = self.hidden_embed(hidden)
        hidden_embed = self.relu(hidden_embed)
        output_time = self.time_linear(hidden_embed).squeeze()
        output_event = self.event_linear(hidden_embed)
        return output_time, output_event

    def loss(self, pred_time, pred_event, gt_time, gt_event):
        time_loss = -(pred_time + self.w * gt_time + self.b +
                      (torch.exp(pred_time + self.b) - torch.exp(pred_time + self.w * gt_time + self.b)) / self.w).mean()

        event_loss = F.cross_entropy(pred_event, gt_event)

        return self.alpha * time_loss + event_loss

    def predict(self, input_time, input_event):
        output_time, output_event = self.forward(input_time, input_event)
        pred_event = torch.argmax(output_event, dim=1)
        output_time = output_time.detach().cpu().numpy()
        w = self.w.detach().cpu().item()
        b = self.b.detach().cpu().item()

        pred_time = torch.tensor([
            quad(lambda t: t * np.exp(vh + w * t + b + (np.exp(vh + b) - np.exp(vh + w * t + b)) / w), a=0, b=10.0)[0]
            for vh in output_time])

        return pred_time, pred_event


class INTPP(nn.Module):
    def __init__(self):
        super(INTPP, self).__init__()
        self.event_classes = cfg.EVENT_CLASSES
        self.event_embed_dim = cfg.EVENT_EMBED_DIM
        self.lstm_hidden_dim = cfg.LSTM_HIDDEN_DIM
        self.hidden_dim = cfg.HIDDEN_DIM
        self.alpha = cfg.ALPHA

        self.event_embed = nn.Embedding(self.event_classes, self.event_embed_dim)
        self.lstm = nn.LSTM(self.event_embed_dim + 1, self.lstm_hidden_dim)
        self.hidden_embed = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.time_linear = nn.Linear(self.hidden_dim, self.event_classes)
        self.event_linear = nn.Linear(self.hidden_dim, self.event_classes)
        self.c = nn.Parameter(torch.ones(self.event_classes) * 0.8)
        self.w = nn.Parameter(-torch.tensor(0.2))

    def forward(self, input_time, input_event):
        event_embed = self.event_embed(input_event)
        time = input_time.unsqueeze(-1)
        lstm_input = torch.cat([time, event_embed], dim=-1)
        hidden, _ = self.lstm(lstm_input)
        hidden = hidden[:, -1, :]
        hidden_embed = self.hidden_embed(hidden)
        hidden_embed = self.relu(hidden_embed)
        output_time = self.time_linear(hidden_embed)
        output_event = self.event_linear(hidden_embed)
        return output_time, output_event

    def loss(self, pred_time, pred_event, gt_time, gt_event):
        c = torch.abs(self.c)

        lj = pred_time

        wdt = gt_time.unsqueeze(-1) * self.w

        lam = torch.exp(lj + wdt) + c

        LAM = (torch.exp(lj + wdt) - torch.exp(lj)) / self.w + torch.matmul(gt_time.unsqueeze(-1), c.unsqueeze(0))

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
        output_time, output_event = self.forward(input_time, input_event)
        pred_event = torch.argmax(output_event, dim=1)
        output_time = output_time.detach().cpu().numpy()
        w = self.w.detach().cpu().item()
        b = self.b.detach().cpu().item()

        pred_time = torch.tensor([
            quad(lambda t: t * np.exp(vh + w * t + b + (np.exp(vh + b) - np.exp(vh + w * t + b)) / w), a=0.0, b=10.0)[0]
            for vh in output_time])

        return pred_time, pred_event

    def get_parameters(self):
        return torch.abs(self.c).detach(), self.w.detach()



