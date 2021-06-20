import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

MAX_LENGTH = 82
device = "cpu"
PAD_IDX = 1


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, n_hid_out, dropout=0.3,
                 n_directions_out=1, n_layers_out=1, max_length=MAX_LENGTH):
        super(TransformerModel, self).__init__()
        self.n_directions_out = n_directions_out
        self.n_layers = n_layers_out
        self.hid_dim = n_hid_out
        self._out_dim = n_directions_out * n_layers_out * n_hid_out
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.out_h = nn.Linear(ninp * MAX_LENGTH, self._out_dim)
        self.out_c = nn.Linear(ninp * MAX_LENGTH, self._out_dim)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.out_h.bias.data.zero_()
        self.out_h.weight.data.uniform_(-initrange, initrange)
        self.out_c.bias.data.zero_()
        self.out_c.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output_flatten = output.view(src.shape[1], -1)
        hidden = self.out_h(output_flatten).reshape(self.n_directions_out * self.n_layers,
                                            src.shape[1], self._out_dim // (self.n_directions_out * self.n_layers))
        cell = self.out_c(output_flatten).reshape(self.n_directions_out * self.n_layers,
                                            src.shape[1], self._out_dim // (self.n_directions_out * self.n_layers))
        hidden = torch.tanh(hidden)
        cell = torch.tanh(cell)
        return output, hidden, cell


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DecoderAttn(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, max_length=MAX_LENGTH):
        super(DecoderAttn, self).__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.attn = nn.Linear(self.hid_dim + self.emb_dim, self.max_length)
        self.attn_combine = nn.Linear(self.hid_dim, self.emb_dim)

        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # print(attn_weights.shape, encoder_outputs.shape, embedded.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0).permute(1, 0, 2),
                                 encoder_outputs.permute(1, 0, 2)
                                 ).permute(1, 0, 2)
        # print(embedded.shape, attn_applied.shape, self.attn_combine)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, src_mask, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_mask)
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs
