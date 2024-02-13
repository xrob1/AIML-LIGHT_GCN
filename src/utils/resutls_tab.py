from src.loader.paths import *
from src import *

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
    
    if dataset_name == YAHOO:
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['verbose']= True
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['save_recs']= True
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['validation_rate']= 27
        configuration['experiment']['models']['external.LightGCN_Custom']['lr']= 0.0014217965357751648
        configuration['experiment']['models']['external.LightGCN_Custom']['epochs']= 27
        configuration['experiment']['models']['external.LightGCN_Custom']['batch_size']= 256
        configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.05948528558207626
        configuration['experiment']['models']['external.LightGCN_Custom']['n_layers']= 3
        configuration['experiment']['models']['external.LightGCN_Custom']['seed']= 123
    if dataset_name == FACEBOOK:
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['verbose']= True
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['save_recs']= True
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['validation_rate']= 2
        configuration['experiment']['models']['external.LightGCN_Custom']['lr']= 0.0028462729478462134
        configuration['experiment']['models']['external.LightGCN_Custom']['epochs']= 2
        configuration['experiment']['models']['external.LightGCN_Custom']['batch_size']= 64
        configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.06184015598288455
        configuration['experiment']['models']['external.LightGCN_Custom']['n_layers']= 3
        configuration['experiment']['models']['external.LightGCN_Custom']['seed']= 123
        
    return configuration

def set_runtime_metrics_configuration(configuration,dataset_name):
    configuration['experiment']['data_config']['train_path'] = dataset_filepath(dataset_name, 'train')
    configuration['experiment']['data_config']['validation_path'] = dataset_filepath(dataset_name, 'val')
    configuration['experiment']['data_config']['test_path'] = dataset_filepath(dataset_name, 'test')
    configuration['experiment']['models']['RecommendationFolder']['folder'] = get_recs_path(dataset_name)#os.path.abspath('data_dz/' + DATASET + '/' + type)
    configuration['experiment']['dataset'] = dataset_name
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
