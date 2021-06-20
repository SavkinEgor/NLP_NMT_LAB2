import random

from nltk.tokenize import WordPunctTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy.data import Field

MAX_LENGTH = 82
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 1

tokenizer_W = WordPunctTokenizer()


def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


SRC = Field(tokenize=tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False, SRC=SRC):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        weight = torch.FloatTensor(SRC.vocab.vectors)
        self.embedding = nn.Embedding.from_pretrained(weight)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
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

    def init_weights(self):
        initrange = 0.1
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.weight_hh_l0.data.uniform_(-initrange, initrange)


class DecoderAttn(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False, max_length=MAX_LENGTH, TRG=TRG):
        super(DecoderAttn, self).__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.max_length = max_length

        weight = torch.FloatTensor(TRG.vocab.vectors)
        self.embedding = nn.Embedding.from_pretrained(weight)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        multiplier = 2 if bidirectional else 1
        self.attn = nn.Linear(self.hid_dim + self.emb_dim, self.max_length)
        self.attn_combine = nn.Linear(self.hid_dim * multiplier + self.emb_dim, self.emb_dim)

        self.out = nn.Linear(
            in_features=hid_dim * multiplier,
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

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # print(output.shape, self.attn_combine)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        # print(output.shape, self.out)
        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def init_weights(self):
        initrange = 0.1
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.weight_hh_l0.data.uniform_(-initrange, initrange)
        self.attn.bias.data.zero_()
        self.attn.weight.data.uniform_(-initrange, initrange)
        self.attn_combine.bias.data.zero_()
        self.attn_combine.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)


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
