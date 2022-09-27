import glob
import os
import random
import re
import string
from collections import Counter
from os import listdir

import keras
import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential

from common.Configuration import Configuration
from common.commonsLib import loggerElk
from text_class.Classifier import Classifier
from text_class.LoadAnimation import Loader


class SeqBinaryClassifier(Classifier):
    logger = loggerElk(__name__)

    def __init__(self):
        conf = Configuration()
        super().__init__(conf.lstm_name, conf)
        self.TOP_WORDS = None  # 10000
        self.DOC_SIZE = 300
        self.VEC_SIZE = 300
        self.lexicon = None
        self.OUTPUT_SIZE = 2
        self.CLASS_TARGET = 1
        self.logger.Information(f'downloading stopwords')
        nltk.download('stopwords')
        self.tokenizer = os.path.join(
            self.conf.working_path, self.conf.lstm_model_path, self.conf.lstm_name + '.tokenizer')
        self.vocab = os.path.join(
            self.conf.working_path, self.conf.lstm_model_path, self.conf.lstm_name + '.analysis.vocab')

    # turn a doc into clean tokens
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

    def add_doc_to_vocab(self, filename, vocab):
        # load doc
        doc = self.load_doc(filename)
        # clean doc
        tokens = self.clean_doc2tokens(doc)
        # update counts
        vocab.update(tokens)

    # load all docs in a directory
    def process_docs(self, directory, vocab, class_id):
        documents = list()
        # walk through all files in the folder
        for filename in listdir(directory):
            # create the full path of the file to open
            path = os.path.join(directory, filename)
            # load the doc
            doc = self.load_doc(path)
            # clean doc
            tokens = self.clean_doc(doc, vocab)
            # add to list
            class_mark = 1 if class_id == self.CLASS_TARGET else 0
            documents.append((
                tokens,
                keras.utils.to_categorical(class_mark, num_classes=self.OUTPUT_SIZE)))

        return documents

    def listDirTextDeep(self, class_name, vocab):
        base = [x for x in os.walk(class_name)][0]
        cwd = os.getcwd()
        subdir = []
        for sub_dir in base[1]:
            directory = os.path.join(base[0], sub_dir)
            subdir.append(directory)
            list_dir = os.listdir(directory)
            random.shuffle(list_dir)
            for file in list_dir[:self.conf.lstm_max_data_set]:
                if file.endswith('.txt'):
                    file_path = os.path.join(directory, file)
                    self.add_doc_to_vocab(file_path, vocab)

        os.chdir(cwd)
        # keep tokens with a min occurrence
        min_occur = 2
        tokens = [k for k, c in vocab.items() if c >= min_occur]
        self.save_list(tokens, self.vocab)
        tokens = set(tokens)
        train_docs = []
        class_list = np.arange(0, len(subdir))
        for class_id, directory in enumerate(subdir):
            # load all training reviews
            it_train_data = self.process_docs(directory, tokens, class_id)
            train_docs.extend(it_train_data)

        return tokens, train_docs

    # save list to file
    @staticmethod
    def save_list(lines, filename):
        # convert lines to a single blob of text
        data = '\n'.join(lines)
        # open file
        file = open(filename, 'w+', encoding='utf-8', errors='surrogateescape')
        # write text
        try:
            file.write(data)
        except Exception:
            pass
        # close file
        file.close()

    @staticmethod
    def clean_doc(doc, lexicon):
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # filter out tokens not in vocab
        tokens = [w.lower() for w in tokens if w.lower() in lexicon]
        tokens = ' '.join(tokens)
        return tokens

    @staticmethod
    def load_vocabulary(vocab_path):
        vocab = SeqBinaryClassifier.load_doc(vocab_path)
        vocab = vocab.split()
        vocab = set(vocab)
        return vocab

    @staticmethod
    def load_doc(filename):
        # open the file as read only
        file = open(filename, 'r', encoding='ISO-8859-1', errors='surrogateescape')

        # read all text
        try:
            text = file.read()
        except Exception:
            text = ''

        # close the file
        file.close()

        return text

    @staticmethod
    def listDirByExt(class_name, ext):
        cwd = os.getcwd()
        os.chdir(class_name)
        file_list = []
        for file in glob.glob("*." + ext):
            file_list.append(file)
        os.chdir(cwd)

        return file_list

    def preprocess(self, class_name):
        vocab = Counter()
        with Loader("Loading dataset"):
            tokens, X = self.listDirTextDeep(class_name, vocab)

        X, y = zip(*X)

        # create the tokenizer
        tokenizer = Tokenizer()

        # fit the tokenizer on the documents
        tokenizer.fit_on_texts(X)

        # sequence encode
        encoded_docs = tokenizer.texts_to_sequences(X)
        self.TOP_WORDS = len(tokenizer.word_index) + 1
        X = pad_sequences(encoded_docs, maxlen=self.DOC_SIZE, padding='post')
        self.save_model2(self.tokenizer, tokenizer)

        return X, y

    def preprocess2predict(self, ocr_text, vocabulary_path=None):
        self.logger.Information(f'LSTM preprocess2predict: load tokenizer')
        tokenizer = self.load_model(self.tokenizer)
        self.TOP_WORDS = len(tokenizer.word_index) + 1
        X = []
        pages = []
        if len(ocr_text) > 0:
            # self.logger.Information('check_report: splitting text by page')
            pages = re.compile(r'\[\[\[\d+\]\]\]').split(ocr_text)

        # load the vocabulary
        vocabulary_path = self.vocab if vocabulary_path is None else vocabulary_path
        vocab = self.load_doc(vocabulary_path)
        vocab = vocab.split()
        self.lexicon = set(vocab)

        for i in range(1, len(pages)):
            X.append(SeqBinaryClassifier.clean_doc(pages[i], self.lexicon))

        encoded_docs = tokenizer.texts_to_sequences(X)
        X = pad_sequences(encoded_docs, maxlen=self.DOC_SIZE, padding='post')

        return X

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Keras LSTM model with Word Embeddings
        model = Sequential()
        model.add(Embedding(self.TOP_WORDS, self.VEC_SIZE, input_length=self.DOC_SIZE))
        model.add(LSTM(300, return_sequences=True))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.OUTPUT_SIZE, activation='sigmoid'))
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train,
                            y_train,
                            epochs=self.conf.lstm_epochs,
                            batch_size=self.conf.lstm_batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[self.tensorboard])
        # print(history)

        evaluation = model.evaluate(X_test, y_test)
        print('Test loss:', evaluation[0])
        print('Test accuracy:', evaluation[1])
        print('Saving', os.path.join(self.conf.working_path, self.conf.lstm_model_path, self.name + '.h5'))
        model.save(os.path.join(self.conf.working_path, self.conf.lstm_model_path, self.name + '.h5'))

        # self.plot_history(self.conf.lstm_model_path, history)

    def predict(self, x):
        self.logger.Information(f'LSTM predict')
        # Keras LSTM model with Word Embeddings
        model = Sequential()
        model.add(Embedding(self.TOP_WORDS, self.VEC_SIZE, input_length=self.DOC_SIZE))
        model.add(LSTM(300, return_sequences=True))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.OUTPUT_SIZE, activation='sigmoid'))
        # print(model.summary())

        self.logger.Information(f'LSTM predict: model compile')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.logger.Information(f'LSTM predict: load weights')
        model.load_weights(os.path.join(
            self.conf.working_path, self.conf.lstm_model_path, self.conf.lstm_name + '.h5'))

        self.logger.Information(f'LSTM predict: do prediction')
        probs = model.predict(x)
        y_pred = np.around(probs)

        return y_pred
