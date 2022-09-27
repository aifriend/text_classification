import os
import random
import re
import string
from collections import Counter
from pprint import pprint

import unicodedata
from nltk.corpus import stopwords

os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['LIBRARIES_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
from pandas import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from common.ClassFile import ClassFile
from common.ExtractorService import ExtractorService


class GbcNlpService:
    DEVICE_TYPE = 'cuda'
    SOURCE = ''
    DF_CONTENT = 'Content'
    DF_CATEGORY = 'Category'
    _LOAD_MODEL_ROOT = './data_volume/.retrain.ns'
    DATA_DIR = r'./dataset/train'
    MODEL_DIR = r'./data_volume'
    MAX_DOC_NUMBER = 9999
    VOCAB_LIB = f'{_LOAD_MODEL_ROOT}.v2.model.vocab'
    LOAD_PATH = f''
    SAVE_PATH = f'{_LOAD_MODEL_ROOT}.v2.model'
    VALIDATION_SIZE = 0.15
    LEARNING_RATE = 1e-6
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    CONTENT_EXT = "txt"

    # BETO Bert spanish pre-trained
    # 'dccuchile/bert-base-spanish-wwm-uncased'
    # 'mrm8488/RuPERTa-base'
    BERT_MODEL = 'dccuchile/bert-base-spanish-wwm-uncased'

    def __init__(self):
        self.epochs = self.NUM_EPOCHS

        self.df = None
        self.label_dict = {}
        self.vocab = ''
        self.dataset_train = None
        self.dataset_val = None
        self.model = None
        self.data_loader_train = None
        self.data_loader_validation = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device(self.DEVICE_TYPE)
        print(f"Device: {self.device}")

    def _loader(self, path, label, c_list, l_list):
        doc_list = ClassFile.list_files_like(path, self.CONTENT_EXT)
        for doc in doc_list[:self.MAX_DOC_NUMBER]:
            doc_text = ClassFile.get_text(doc)
            doc_text = ExtractorService.clean_doc(doc_text)
            doc_text = ExtractorService.simplify(doc_text)
            c_list.append(doc_text)
            l_list.append(label)

        return c_list, l_list

    def load_data(self):
        # read dataset from TXT
        cont_list = list()
        label_list = list()

        # train
        # read dataset from directory
        doc_class_1_path = os.path.join(self.DATA_DIR, "NS")
        cont_list, label_list = self._loader(doc_class_1_path, "NS", cont_list, label_list)
        doc_class_path = os.path.join(self.DATA_DIR, "OTR")
        cont_list, label_list = self._loader(doc_class_path, "OTR", cont_list, label_list)

        # build vocabulary
        self.vocab = NlpTool.get_vocabulary(self.DATA_DIR, self.MODEL_DIR, f"{self.VOCAB_LIB}")

        # build dataframe
        print("Loading from directory...")
        self.df = pd.DataFrame(list(zip(cont_list, label_list)), columns=['Content', 'Category'])
        # read dataset from CSV
        # print(f"Loading from {self.SOURCE}...")
        # self.df = pd.read_csv(self.SOURCE, skip_blank_lines=True, delimiter="#")

        print(f"\n{self.df.head()}")

        possible_labels = self.df.Category.unique()
        self.label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            self.label_dict[possible_label] = index
        print(f"\nClasses: {self.label_dict}")

        # add label_text by numbers
        self.df['label'] = self.df.Category.replace(self.label_dict)

        """
        Because the labels are imbalanced,
        we split the data set in a stratified fashion,
        using this as the class labels.
        """
        X_train, X_val, y_train, y_val = train_test_split(self.df.index.values,
                                                          self.df.label.values,
                                                          test_size=self.VALIDATION_SIZE,
                                                          random_state=42,
                                                          stratify=self.df.label.values)

        self.df['data_type'] = ['not_set'] * self.df.shape[0]
        self.df.loc[X_train, 'data_type'] = 'train'
        self.df.loc[X_val, 'data_type'] = 'val'
        print(f"\n{self.df.groupby(['Category', 'label', 'data_type']).count()}\n")

        # max token length
        print(f"Clean dataset content for {len(self.df['Content'])}")
        self.df["Content"] = self.df["Content"].map(lambda x: NlpTool.clean(x, self.vocab))
        print(f"\n{self.df.sample(10)}")

    def tokenizer(self, mode="train"):
        """
        Tokenization is a process to take raw texts and split into tokens, which are numeric data to represent words.
        Constructs a BERT tokenizer. Based on WordPiece.
        Instantiate a pre-trained BERT model configuration to encode our data.
        To convert all the titles from text into encoded form, we use a function called batch_encode_plus,
        We will proceed train and validation data separately.
        The 1st parameter inside the above function is the title text.
        add_special_tokens=True means the sequences will be encoded with the special tokens relative to their model.
        When batching sequences together, we set return_attention_mask=True,
        so it will return the attention mask according to the specific tokenizer defined by the max_length attribute.
        We also want to pad all the titles to certain maximum length.
        We actually do not need to set max_length=256, but just to play it safe.
        return_tensors='pt' to return PyTorch.
        And then we need to split the data into input_ids, attention_masks and labels.
        Finally, after we get encoded data set, we can create training data and validation data.
        """
        print(f"\nLoading BERT tokenizer")
        tokenizer = BertTokenizer.from_pretrained(
            self.BERT_MODEL,
            use_fast=False,
            strip_accents=True,
            do_lower_case=True)
        torch.save(tokenizer, self.SAVE_PATH + ".tokenizer")

        # encoding
        if mode == "train":
            encoded_data_train = tokenizer.batch_encode_plus(
                self.df[self.df.data_type == 'train'].Content.values,
                add_special_tokens=True,
                return_attention_mask=True,
                padding='longest',
                return_tensors='pt'
            )
            pprint(encoded_data_train)
            # seq_len = [len(i.split()) for i in self.df.Content.values]
            # pd.Series(seq_len).plot.hist(bins=30)
            # train_text.tolist(),
            # max_length = 25,
            # pad_to_max_length = True,
            # truncation = True

            # to tensor dataset
            input_ids_train = encoded_data_train['input_ids']
            attention_masks_train = encoded_data_train['attention_mask']
            labels_train = torch.tensor(self.df[self.df.data_type == 'train'].label.values)
            self.dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

            # Data Loaders
            self.data_loader_train = DataLoader(self.dataset_train,
                                                sampler=RandomSampler(self.dataset_train),
                                                batch_size=self.BATCH_SIZE)

        encoded_data_val = tokenizer.batch_encode_plus(
            self.df[self.df.data_type == 'val'].Content.values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='longest',
            return_tensors='pt'
        )
        # pprint(encoded_data_val)

        # to tensor dataset
        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(self.df[self.df.data_type == 'val'].label.values)
        self.dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

        # Data Loaders
        self.data_loader_validation = DataLoader(self.dataset_val,
                                                 sampler=SequentialSampler(self.dataset_val),
                                                 batch_size=self.BATCH_SIZE)

    def make_model(self, mode="train"):
        """
        We are treating each text as its unique sequence, so one sequence will be classified to one labels
        "model/beto_pytorch_uncased" is a smaller pre-trained model.
        Using num_labels to indicate the number of output labels.
        We don’t really care about output_attentions.
        We also don’t need output_hidden_states.
        DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset.
        We use RandomSampler for training and SequentialSampler for validation.
        Given the limited memory in my environment, I set batch_size=3.
        """
        if not self.LOAD_PATH:
            print(f"\nLoading BERT model: {self.BERT_MODEL}")
            self.model = BertForSequenceClassification.from_pretrained(
                self.BERT_MODEL,
                num_labels=len(self.label_dict),
                output_attentions=False,
                output_hidden_states=False)
        else:
            try:
                self.model = torch.load(self.LOAD_PATH + ".all", map_location=self.device)
                print(f"\nLoading pre-trained model: {self.LOAD_PATH}.all")
            except Exception as _:
                print(f"\nFail Loading from {self.LOAD_PATH}... trying load BERT model: {self.BERT_MODEL}")
                self.model = BertForSequenceClassification.from_pretrained(
                    self.BERT_MODEL,
                    num_labels=len(self.label_dict),
                    output_attentions=False,
                    output_hidden_states=False)

        self.model.to(self.DEVICE_TYPE)

        if mode == "train":
            """
            To construct an optimizer, we have to give it an iterable containing the parameters to optimize.
            Then, we can specify optimizer-specific options such as the learning rate, epsilon, etc.
            I found epochs=5 works well for this data set.
            Create a schedule with a learning rate that decreases linearly from the initial learning rate
                set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to
                the initial learning rate set in the optimizer.
            """
            self.optimizer = Adam(self.model.parameters(),
                                  lr=self.LEARNING_RATE,
                                  eps=1e-8)

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.data_loader_train) * self.epochs)

    @staticmethod
    def _f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def _accuracy_per_class(self, preds, labels):
        label_dict_inverse = {v: k for k, v in self.label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}'
                  f' -> {round(len(y_preds[y_preds == label]) / len(y_true), 2)}\n')

    def _evaluate(self):
        self.model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in self.data_loader_validation:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(self.data_loader_validation)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    def train(self):
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        best_val_f1 = 0.0
        for epoch in tqdm(range(1, self.epochs + 1)):
            # set train mode
            self.model.train()

            loss_train_total = 0

            progress_bar = tqdm(
                self.data_loader_train, desc='Epoch {:1d}'.format(epoch),
                leave=False, position=0, disable=False)
            for batch in progress_bar:
                self.model.zero_grad()

                batch = tuple(b.to(self.device) for b in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2],
                          }

                outputs = self.model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            loss_train_avg = loss_train_total / len(self.data_loader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = self._evaluate()
            tqdm.write(f'Validation loss: {val_loss}')

            val_f1 = self._f1_score_func(predictions, true_vals)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.SAVE_PATH)
                torch.save(self.model, self.SAVE_PATH + ".all")
                tqdm.write(f'Saving best F1 Score: {best_val_f1}')

            if val_loss < 0.10 and val_f1 >= 0.987:
                tqdm.write(f'\nFinal F1 Score: {best_val_f1}')
                print(f"\nLoading best-trained model: {self.SAVE_PATH}.all")
                break

    def predict(self):
        self.model.eval()
        _, predictions, true_vals = self._evaluate()
        self._accuracy_per_class(predictions, true_vals)


class NlpTool:
    MAX_DOC_LENGTH = 400

    @staticmethod
    def clean_doc2tokens(doc):
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word.lower() for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('spanish'))
        tokens = [word for word in tokens if word not in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        return tokens

    @staticmethod
    def add_doc_to_vocab(file_name, vocab):
        # load doc
        doc = ClassFile.get_text(file_name)
        # clean doc
        tokens = NlpTool.clean_doc2tokens(doc)
        # update counts
        vocab.update(tokens)

        return tokens

    @staticmethod
    def get_vocabulary(data_path, vocab_path, vocab_load_name):
        print(f"Load vocabulary from {vocab_path}...")
        if os.path.isfile(vocab_load_name):
            tokens = NlpTool.get_vocabulary_from_file(vocab_load_name)
            if tokens is not None:
                return tokens
            else:
                raise ValueError(f"Vocabulary '{vocab_load_name}' is not found!")

        vocab = Counter()
        base = [x for x in os.walk(data_path)][0]
        cwd = os.getcwd()
        subdir = []
        load_count = 0
        for sub_dir in base[1]:
            directory = os.path.join(base[0], sub_dir)
            subdir.append(directory)
            for file in os.listdir(directory):
                if file.endswith(rf".{GbcNlpService.CONTENT_EXT}"):
                    file_path = os.path.join(directory, file)
                    NlpTool.add_doc_to_vocab(file_path, vocab)
                    if load_count % 100 == 0:
                        print(">", end="")
                    load_count += 1
        print()

        # keep tokens with a min occurrence
        os.chdir(cwd)
        min_occur = 2
        tokens = [k for k, c in vocab.items() if c >= min_occur]
        table = str.maketrans('', '', string.punctuation)
        tokens = sorted(list(set([w.translate(table) for w in tokens])))
        if not os.path.isfile(f"{vocab_load_name}"):
            print(f"Save vocabulary to {vocab_load_name}...")
            for token in tokens:
                ClassFile.to_txtfile(
                    data=f"{token}\n", file_=f"{vocab_load_name}", mode="+a")

        return tokens

    @staticmethod
    def get_vocabulary_from_file(vocab_path):
        if os.path.isfile(vocab_path):
            token_from = sorted(ClassFile.get_content(f"{vocab_path}"))
            return token_from

        return None

    @staticmethod
    def clean(content, vocab_list):
        content = NlpTool.clean_text(content)
        content = NlpTool.simplify(content)

        # remove punctuation from each token
        tokens = content[:NlpTool.MAX_DOC_LENGTH].split()
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [w.lower() for w in tokens if w.lower() in vocab_list]
        content_tokens = ' '.join(tokens)
        print(".", end="")

        return content_tokens

    @staticmethod
    def clean_text(text):
        if text is None:
            return ''

        new_ = re.sub(r'\n', ' ', text)
        new_ = re.sub(r'\n\n', r'\n', new_)
        new_ = re.sub(r'\\n', ' ', new_)
        new_ = re.sub(r'\t', ' ', new_)

        # page number
        new_ = re.sub(r'(\[\[\[.{1,3}\]\]\])', '', new_)
        new_ = re.sub(r'\- F.?o.?l.?i.?o \d{1,2} \-', '', new_)
        new_ = re.sub(r"F.?o.?l.?i.?o.{0,2}-.{0,2}[0-9]{1,2}.{0,2}-", '', new_)

        new_ = re.sub(r'(\d{5})-(\W*)(\w{3,20}?)\W', r'\1 \2', new_)

        new_ = re.sub(r'--', r'-', new_)
        new_ = re.sub(r'=', ' ', new_)
        new_ = re.sub(r':', ' ', new_)
        new_ = re.sub(r'\( ', r'(', new_)
        new_ = re.sub(r' \)', r')', new_)
        new_ = re.sub(r'"', ' ', new_)
        new_ = re.sub(r'_', ' ', new_)
        new_ = re.sub(r'\'', ' ', new_)
        new_ = re.sub(r'/', ' ', new_)

        new_ = re.sub(r'(\s[a-z0-9]{1,3})-([a-z0-9]{1,3}\s)', r'\1 \2', new_)
        new_ = re.sub(r'\s{2,1000}', " ", new_)
        new_ = re.sub(r'\((\d{5})\)', r'\1', new_)  # inside curly brackets
        new_ = re.sub(r'\((?!CP)[.\w\s\-%]{1,100}\)', r' ', new_)  # inside curly brackets but not CP for postal code
        new_ = re.sub(r"(\w{2})-\s(\w{2})", r'\1\2', new_)  # join new line words

        return new_.strip()

    @staticmethod
    def simplify(text):
        clean_text = text
        try:
            accents = ('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT', 'COMBINING TILDE')
            accents = set(map(unicodedata.lookup, accents))
            chars = [c for c in unicodedata.normalize('NFD', text) if c not in accents]
            if chars:
                clean_text = unicodedata.normalize('NFC', ''.join(chars))
        except NameError:
            pass

        return str(clean_text)


def main():
    text_service = GbcNlpService()

    print(f"\nPre-process data from {GbcNlpService.DATA_DIR}...")
    text_service.load_data()

    print(f"\nTokenizing data...")
    text_service.tokenizer(mode="train")

    print(f"\nCreate model from {GbcNlpService.BERT_MODEL}...")
    text_service.make_model(mode="train")

    print(f"\nTraining...")
    text_service.train()

    print(f"\nPrediction...")
    text_service.predict()


if __name__ == '__main__':
    main()
