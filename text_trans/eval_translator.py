import math
import os
import re
from typing import Optional

import spacy
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
from torchtext import data

DEVICE_TYPE = 'cuda'
device = torch.device(DEVICE_TYPE)
print(f"Device: {device}")

VERSION = "1_0"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MyTransformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", _source_vocab_length: int = 60000,
                 _target_vocab_length: int = 60000) -> None:
        super(MyTransformer, self).__init__()
        self.source_embedding = nn.Embedding(_source_vocab_length, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.target_embedding = nn.Embedding(_target_vocab_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.out = nn.Linear(512, _target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src.size()[1] != tgt.size()[1]:
            raise RuntimeError("the batch number of src and tgt must be equal")
        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.out(output)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


# Load the Spacy Models- These will be used for tokenization of german and english text.
spacy_es = spacy.load("es_core_news_sm")


def tokenize_es(text):
    return [tok.text for tok in spacy_es.tokenizer(text)]


def greedy_decode_sentence(_model, _sentence):
    _model.eval()
    _sentence = SRC.preprocess(_sentence)
    indexed = []
    for tok in _sentence:
        if SRC.vocab.stoi[tok] != 0:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(0)
    _sentence = Variable(torch.LongTensor([indexed])).cuda()
    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]]).cuda()
    translated_sentence = ""
    maxlen = 25
    for i in range(maxlen):
        size = trg.size()[0]
        np_mask = torch.triu(torch.ones(size, size)).transpose(0, 1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1,
                                                                                       float(0.0))
        np_mask = np_mask.cuda()
        pred = _model(_sentence.transpose(0, 1), trg, tgt_mask=np_mask)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]
        translated_sentence += " " + add_word
        if add_word == EOS_WORD:
            break
        trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        # print(trg)

    return translated_sentence


def dump(doc):
    _file_path = os.path.dirname(os.path.realpath(__file__)) + "/.data/ttn_result.csv"
    with open(_file_path, "a") as output:
        output.write(doc + "\n")


if __name__ == '__main__':
    # Define some special tokens we will use for specifying blank/padding words, and beginning and end of sentences.
    BOS_WORD = '<sos>'
    EOS_WORD = '<eos>'

    # We start by defining a preprocessing pipeline for both our source and target sentence
    SRC = data.Field(tokenize=tokenize_es)
    TGT = data.Field(tokenize=tokenize_es, init_token=BOS_WORD, eos_token=EOS_WORD)

    # We then use the implemented function splits to divide our datasets into train,validation and test datasets.
    # We also filter our sentences using the max_len parameter so that our code runs a lot faster.
    data_fields = [('src', SRC), ('trg', TGT)]
    train, val, test = data.TabularDataset.splits(
        path='.data/', fields=data_fields,
        train='train.csv', validation='val.csv', test='test.csv',
        format='csv')

    # We also create a Source and Target Language vocabulary by using the built in function in data field object.
    # We also specify a MIN_FREQ of 2 so that any word that doesn't occur at least twice doesn't get to be a part
    # of our vocabulary.
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    # Just load model for inference
    model = MyTransformer(
        _source_vocab_length=source_vocab_length, _target_vocab_length=target_vocab_length)
    model = model.cuda()
    model.load_state_dict(torch.load(f"checkpoint/checkpoint_best_v{VERSION}.pt"))
    model.eval()

    for i, example in enumerate(test):
        print(f"  Original: {' '.join(example.src)}")
        predicted = greedy_decode_sentence(model, example.src)
        predicted = re.sub(rf"{EOS_WORD}", '', predicted, count=1)
        print(f"Translated: {predicted.strip()}")
        print(f"    Target: {' '.join(example.trg)}")
        print("---")
        #dump(' '.join(example.src) + ";" + predicted)
        if i > 30:
            break
