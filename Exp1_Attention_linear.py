import random

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_LENGTH = 82
device = "cpu"
PAD_IDX = 1


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim,
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        inp = torch.ones(MAX_LENGTH, src.shape[1], dtype=torch.long, device=device) * PAD_IDX
        inp[:src.shape[0], :] = src
        embedded = self.embedding(inp)

        embedded = self.dropout(embedded)

        output, (hidden, cell) = self.rnn(embedded)

        return output, hidden, cell


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
        self.attn_combine = nn.Linear(self.hid_dim + self.emb_dim, self.emb_dim)

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
        attn_applied = torch.bmm(attn_weights.unsqueeze(0).permute(1, 0, 2),
                                 encoder_outputs.permute(1, 0, 2)
                                 ).permute(1, 0, 2)

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

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs
