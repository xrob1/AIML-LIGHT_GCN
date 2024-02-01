from src.loader.paths import *

def set_dataset_configuration(configuration: dict, dataset_name: str):
    """
    Given a dataset adds to the basic configuration file the dataset information
    :param configuration: dictionary containing the structure of the YAML config file
    :param dataset_name: name of the dataset
    :return: modified configuration dict
    """
    configuration = dict(configuration)
    configuration['experiment']['data_config']['train_path'] = dataset_filepath(dataset_name, 'train')
    configuration['experiment']['data_config']['validation_path'] = dataset_filepath(dataset_name, 'val')
    configuration['experiment']['data_config']['test_path'] = dataset_filepath(dataset_name, 'test')
    configuration['experiment']['dataset'] = dataset_name
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['dataset_name'] = dataset_name
    return configuration
