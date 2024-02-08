import os
from src import PROJECT_PATH

DATA_DIR = os.path.join(PROJECT_PATH, 'data')
RESULT_DIR = os.path.join(PROJECT_PATH, 'results')
CONFIG_DIR = os.path.join(PROJECT_PATH, 'config_files')

RAW_DATA_FOLDER = 'data'
MAIN_DIR = [RESULT_DIR]

DATASET_NAME = 'dataset'
TRAIN_NAME = 'train'
VAL_NAME = 'val'
TEST_NAME = 'test'


DATASET_NAME_BY_TYPE = {
    'raw': os.path.join('data', 'dataset.tsv'),
    'processed': f'{DATASET_NAME}.tsv',
    'train': f'{TRAIN_NAME}.tsv',
    'val': f'{VAL_NAME}.tsv',
    'test': f'{TEST_NAME}.tsv',
}

BASE_CONFIGURATION_FILE_NAME = 'custom.yml'
FACEBOOK_EXPLORATION_FILE_NAME ='facebook_exploration.yml'
YHOO_EXPLORATION_FILE_NAME ='yahoo_exploration.yml'

def create_directory(dir_path: str):
    """
    Check that the directory exists, otherwise create it
    @param dir_path: path of the directory to create
    @return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'Created directory at \'{dir_path}\'')


def dataset_directory(dataset_name: str):
    """
    Given the dataset name returns the dataset directory
    @param dataset_name: name of the dataset
    @return: the path of the directory containing the dataset data
    """
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f'Directory at {dataset_dir} not found. Please, check that dataset directory exists')
    return os.path.abspath(dataset_dir)


def dataset_filepath(dataset_name: str, type='raw'):
    """
    Given the dataset name returns the path of the dataset file
    @param dataset_name: name of the dataset
    @param type: type of dataset. Raw, clean, training, validation or test
    @return: the path of the directory containing the dataset data
    """
    assert type in DATASET_NAME_BY_TYPE.keys(), f'Incorrect dataset type. Dataset type found {type}.'
    dataset_dir = dataset_directory(dataset_name)
    filepath = os.path.join(dataset_dir, DATASET_NAME_BY_TYPE[type])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'File at {filepath} not found. Please, check your files')
    return os.path.abspath(filepath)


def basic_conf_file():
    """
    Returns the path of the file containing the basic configuration for Elliot
    @return: the absolute path of the basic configuration file
    """
    config_path = os.path.join(CONFIG_DIR, BASE_CONFIGURATION_FILE_NAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Basic configuration file at {config_path} not found. '
                                f'Please, check that the file exists')
    return os.path.abspath(config_path)

def yahoo_exp_conf_file():
    """
    Returns the path of the file containing the yahoo exploration configuration for Elliot
    @return: the absolute path of the basic configuration file
    """
    config_path = os.path.join(CONFIG_DIR, YHOO_EXPLORATION_FILE_NAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Basic configuration file at {config_path} not found. '
                                f'Please, check that the file exists')
    return os.path.abspath(config_path)

def facebook_exp_conf_file():
    """
    Returns the path of the file containing the facebook book exploration configuration for Elliot
    @return: the absolute path of the basic configuration file
    """
    config_path = os.path.join(CONFIG_DIR, FACEBOOK_EXPLORATION_FILE_NAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Basic configuration file at {config_path} not found. '
                                f'Please, check that the file exists')
    return os.path.abspath(config_path)