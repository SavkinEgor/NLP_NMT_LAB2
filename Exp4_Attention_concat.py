import random

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 82
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 1


class EncoderAttn(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        lstm_dropout = dropout if n_layers > 1 else 0
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers, bidirectional=True, dropout=lstm_dropout)
        self.fc_h = nn.Linear(enc_hid_dim * 2 * n_layers, dec_hid_dim)
        self.fc_c = nn.Linear(enc_hid_dim * 2 * n_layers, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(
            self.fc_h(torch.cat((hidden[-2, :, :],
                                 hidden[-1, :, :],
                                 hidden[-3, :, :],
                                 hidden[-4, :, :]),
                                dim=1)))
        cell = torch.tanh(self.fc_c(torch.cat((cell[-2, :, :],
                                               cell[-1, :, :],
                                               cell[-3, :, :],
                                               cell[-4, :, :]),
                                              dim=1)))
        return outputs, hidden, cell.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        #         print(hidden.shape, encoder_outputs.shape, self.attn)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #         print(hidden.shape, encoder_outputs.shape, self.attn, self.v)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class DecoderAttn(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers
        lstm_dropout = dropout if n_layers > 1 else 0
        self.embedding = nn.Embedding(output_dim, emb_dim)

        #         self.fc_h = nn.Linear(dec_hid_dim, dec_hid_dim * n_layers)

        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, 1, dropout=lstm_dropout)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        #         print(hidden.shape, encoder_outputs.shape, self.attention, "\n\n")
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        #         hidden = self.fc_h(hidden)
        #         hidden = hidden.reshape(self.n_layers, BATCH_SIZE, self.dec_hid_dim)
        #         cell = cell.reshape(self.n_layers, BATCH_SIZE, self.dec_hid_dim)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell))
        #         assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
