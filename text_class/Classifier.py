import os
import pickle
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.saving.model_config import model_from_json

from text_class.ClassImage import ClassImage


class Classifier(object):

    def __init__(self, name, conf):
        self.conf = conf
        self.name = '{}_{}'.format(name, int(time.time()))
        self.model = None
        self.onehot_encoder = None
        self.classes = []
        self.class_dict = None
        self.tensorboard = TensorBoard(log_dir=os.path.join(conf.logs_path, '{}'.format(self.name)),
                                       write_graph=True,
                                       write_images=True,
                                       histogram_freq=1
                                       # profile_batch=1,
                                       # embeddings_freq=1
                                       )
        self.info = {}
        print(self.tensorboard.log_dir)

    def save_keras_model(self, filename):
        """
        save keras model
        :param filename: file name
        :return:
        """
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(filename + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(filename + '.h5')
        print("Saved model to disk")

    def save_model(self, filename):
        """
        save scikit-learn model
        """
        pickle.dump(self.model, open(filename, 'wb'))

    @staticmethod
    def save_model2(filename, model):
        with open(filename, 'wb') as handle:
            pickle.dump(model, handle)  # , protocol=pickle.HIGHEST_PROTOCOL)

    def plot_history(self, path, history, show=False):
        plt.style.use('ggplot')

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join(path, self.name + '.png'))
        if show is True:
            plt.show()

    @staticmethod
    def load_keras_model_s3(conf):
        """
        load keras model
        :param conf: conf with s3service
        :return: loaded model
        """
        filename = conf.model_domain + "/" + conf.emb_cnn_i_model
        # load json and create model
        model_json = conf.model_s3_service.get_txt_file(filename + '.json', decode_base64=False)
        model = model_from_json(model_json)
        # load weights into new model
        # Necesito crear temp file para que tensorflow haga load_weights
        h5_bytes = conf.model_s3_service.get_byte_file(filename + '.h5')
        temp = None
        try:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(h5_bytes)
        finally:
            temp.close()
        if temp:
            print("Start loading model from the S3 resource")
            model.load_weights(temp.name)
            print("Loaded model from S3")
            os.unlink(temp.name)
        return model

    @staticmethod
    def load_keras_model(filename):
        """
        load keras model
        :param filename: file name full path
        :return: loaded model
        """
        # load json and create model
        json_file = open(filename + '.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        # load weights into new model
        model.load_weights(filename + '.h5')
        print("Loaded model from disk")
        return model

    @staticmethod
    def load_model(filename):
        """
        load scikit-learn model
        """
        if os.path.getsize(filename) > 0:
            with open(filename, 'rb') as fp:
                return pickle.load(fp)

    def loadImage(self, path):
        im = Image.open(path).convert('L')
        im = im.resize(self.conf.size)
        # im.show()
        data = np.asarray(im)
        data = data[..., np.newaxis]
        data = np.invert(data)
        return data

    def loadImageBIO(self, path):
        im = Image.open(path).convert('L')
        im = im.resize(self.conf.size_bio)
        # im.show()
        data = np.asarray(im)
        data = data[..., np.newaxis]
        data = np.invert(data)
        return data

    @staticmethod
    def checkGPU():
        print(tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    def preprocess_2_predict(self, filename):
        pages = convert_from_path(filename, dpi=300)
        self.info['numPages'] = len(pages)
        if len(pages) > 0:
            pages[0].save(filename + '.png')

            img = ClassImage.load_image(filename + '.png')
            img = ClassImage.resize_image_loaded(img, 256, 340)
            img = ClassImage.crop_image_loaded(img, 256, 256)

            img = ClassImage.denoise(ClassImage.gray_image(img))
            img = img[..., np.newaxis]
            return img

        return None

    @staticmethod
    def preprocess_2_predict_data(data):
        pages = convert_from_bytes(data, dpi=300)
        if len(pages) > 0:
            img = np.array(pages[0])
            img = ClassImage.resize_image_loaded(img, 256, 340)
            img = ClassImage.crop_image_loaded(img, 256, 256)

            img = ClassImage.invert(ClassImage.denoise(ClassImage.gray_image(img)))
            return img

        return None
