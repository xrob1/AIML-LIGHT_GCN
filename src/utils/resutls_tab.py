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

def set_model_factors(configuration: dict,n_factors: int ):
    """Changes the number of fatctors with wich the model will be built with
        :param configuration: dictionary containing the structure of the YAML config file
        :param n_factors: int number of factors
        :return: modified configuration dict
    """
    configuration['experiment']['models']['external.LightGCN_Custom']['factors'] = n_factors
    return configuration

def set_reducers_configuration(configuration: dict,reducers_types:list,reducers_factors:list,kpca_kernels:list):
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['reducers_types'] = reducers_types
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['reducers_factors'] = reducers_factors
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['KPCA_Kernels'] =kpca_kernels# ['linear', 'poly','rbf', 'sigmoid','cosine']
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['tsne_reducers_factors'] = [2]
    return configuration
