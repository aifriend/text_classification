import os

import yaml


class Configuration:

    def __init__(self, conf_route="tc_config.yml", working='../text_class'):
        self.root_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "./")
        self.working_path = os.path.join(self.root_path, working)
        try:
            with open(os.path.join(self.root_path, conf_route), 'r') as yml_file:
                cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
        except Exception as e:
            print('ERROR - Configuration::__init__::{0}::{1}'.format(conf_route, str(e.args)))
            raise e

        self.logs_path = cfg['logs_path']

        # splitter txt train model
        lstm = cfg['lstm']
        self.lstm_name = lstm['name']
        self.lstm_epochs = lstm['epochs']
        self.lstm_batch_size = lstm['batch_size']
        self.lstm_model_path = lstm['model_path']
        self.lstm_max_data_set = lstm['dataset_size']
