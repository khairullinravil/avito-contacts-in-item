import torch
from torch import nn
from torch.nn import functional as F


class BasicModel(nn.Module):
    def __init__(self, inp_voc, emb_size=128, hid_size=128):
        super().__init__()
        self.inp_voc = inp_voc
        self.hid_size = hid_size
        self.embedding = nn.Embedding(len(inp_voc), emb_size)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.linear = nn.Linear(emb_size, 1)
        self.logits = nn.Sigmoid()
        
    def forward(self, inp):
        inp_emb = self.embedding(inp)
        batch_size = inp.shape[0]
        
        enc_seq, [last_state_but_not_really] = self.gru(inp_emb)

        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]

        return self.logits(self.linear(last_state))