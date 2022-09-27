import math
import time
from typing import Optional

import pandas as pd
import plotly.express as px
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
from torchtext import data
from tqdm import tqdm

DEVICE_TYPE = 'cuda'
device = torch.device(DEVICE_TYPE)
print(f"Device: {device}")

spacy_es = spacy.load("es_core_news_sm")

VERSION_LOAD = ""
NEW_VERSION = "2_0"
LEARNING_RATE = 0.00001
BATCH_SIZE = 400
NUM_EPOC = 250
TEST_SENTENCE = "tres cientos trece metros con veintidos decimetros"

global max_src_in_batch, max_tgt_in_batch


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


def translator_train(_train_iter, _val_iter, _model, _optim, num_epochs, use_gpu=True):
    since = time.time()

    best_acc = 0.0
    _train_losses = []
    _valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0
        _model.train()

        for _batch in tqdm(_train_iter, leave=False, position=0, disable=False):
            src = _batch.src.cuda() if use_gpu else _batch.src
            trg = _batch.trg.cuda() if use_gpu else _batch.trg
            # change to shape (bs , max_seq_len)
            src = src.transpose(0, 1)
            # change to shape (bs , max_seq_len+1) , Since right shifted
            trg = trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask = (src != 0)
            src_mask = src_mask.float().masked_fill(
                src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            src_mask = src_mask.cuda() if use_gpu else src_mask
            trg_mask = (trg_input != 0)
            trg_mask = trg_mask.float().masked_fill(
                trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
            trg_mask = trg_mask.cuda() if use_gpu else trg_mask
            size = trg_input.size(1)
            np_mask = torch.triu(torch.ones(size, size)).transpose(0, 1)
            np_mask = np_mask.float().masked_fill(
                np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask.cuda() if use_gpu else np_mask

            # Forward, backprop, optimizer
            _optim.zero_grad()
            preds = _model(
                # , src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
                src.transpose(0, 1), trg_input.transpose(0, 1), tgt_mask=np_mask)
            preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
            loss = F.cross_entropy(preds, targets, ignore_index=0, reduction='sum')
            loss.backward()
            _optim.step()
            train_loss += loss.item() / BATCH_SIZE

        _model.eval()
        with torch.no_grad():
            for _batch in tqdm(_val_iter, leave=False, position=0, disable=False):
                src = _batch.src.cuda() if use_gpu else _batch.src
                trg = _batch.trg.cuda() if use_gpu else _batch.trg
                # change to shape (bs , max_seq_len)
                src = src.transpose(0, 1)
                # change to shape (bs , max_seq_len+1) , Since right shifted
                trg = trg.transpose(0, 1)
                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)
                src_mask = (src != 0)
                src_mask = src_mask.float().masked_fill(
                    src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
                src_mask = src_mask.cuda() if use_gpu else src_mask
                trg_mask = (trg_input != 0)
                trg_mask = trg_mask.float().masked_fill(
                    trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
                trg_mask = trg_mask.cuda() if use_gpu else trg_mask
                size = trg_input.size(1)
                # print(size)
                np_mask = torch.triu(torch.ones(size, size)).transpose(0, 1)
                np_mask = np_mask.float().masked_fill(
                    np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                preds = _model(
                    # , src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
                    src.transpose(0, 1), trg_input.transpose(0, 1), tgt_mask=np_mask)
                preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
                loss = F.cross_entropy(preds, targets, ignore_index=0, reduction='sum')
                valid_loss += loss.item() / 1

        # Log after each epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] complete. "
            f"Train Loss: {train_loss / len(_train_iter):.3f}. "
            f"Val Loss: {valid_loss / len(_val_iter):.3f}")

        # Save best model till now:
        if valid_loss / len(_val_iter) < min(_valid_losses, default=1e9):
            best_acc = valid_loss / len(_val_iter)
            print("new state dict")

        _train_losses.append(train_loss / len(_train_iter))
        _valid_losses.append(valid_loss / len(_val_iter))

        # Check Example after each epoch:
        print(f"Original Sentence: {TEST_SENTENCE}")
        print(f"Translated Sentence: {greedy_decode_sentence(_model, TEST_SENTENCE)}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return _model, _train_losses, _valid_losses


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
        np_mask = np_mask.float().masked_fill(
            np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda()
        pred = _model(_sentence.transpose(0, 1), trg, tgt_mask=np_mask)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]
        translated_sentence += " " + add_word
        if add_word == EOS_WORD:
            break
        trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        # print(trg)

    return translated_sentence


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

    # Let us try to see what our data looks like:
    for i, example in enumerate([(x.src, x.trg) for x in train[0:10]]):
        print(f"Example_{i}:{example}")

    # We also create a Source and Target Language vocabulary by using the built in function in data field object.
    # We also specify a MIN_FREQ of 2 so that any word that doesn't occur at least twice doesn't get to be a part
    # of our vocabulary.
    MIN_FREQ = 2
    vocab, = data.TabularDataset.splits(
        path='.data/', fields=data_fields, train='vocab.csv', format='csv')
    SRC.build_vocab(vocab.src, min_freq=MIN_FREQ)
    TGT.build_vocab(vocab.trg, min_freq=MIN_FREQ)

    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    model = MyTransformer(
        nhead=16, d_model=1024, dim_feedforward=4096,
        _source_vocab_length=source_vocab_length, _target_vocab_length=target_vocab_length)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    model = model.cuda()
    if VERSION_LOAD:
        print(f"loading state dict checkpoint_best_v{VERSION_LOAD}.pt...")
        model.load_state_dict(torch.load(f"checkpoint/checkpoint_best_v{VERSION_LOAD}.pt"))

    # Once we are done with this we can simply use data. Bucketiterator which is used to giver batches of similar length
    # to get our train iterator and validation iterator. Note that we use a batch_size of 1 for our validation data.
    # Its optional to do this but is actually done so that we don't do padding or do minimal padding while checking
    # validation data performance|
    train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))
    val_iter = data.BucketIterator(val, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))
    # training
    model, train_losses, valid_losses = translator_train(train_iter, val_iter, model, optim, NUM_EPOC)

    # Save best model till now:
    print("saving state dict...")
    torch.save(model.state_dict(), f"checkpoint/checkpoint_best_v{NEW_VERSION}.pt")

    losses = pd.DataFrame({'train_loss': train_losses, 'val_loss': valid_losses})
    print(losses.head())
    px.line(losses, y=['train_loss', 'val_loss'])
