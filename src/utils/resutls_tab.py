from src.loader.paths import *
from src import *

def set_dataset_configuration(configuration: dict, dataset_name: str,tipo:str):
    """
    Given a dataset adds to the basic configuration file the dataset information
    :param configuration: dictionary containing the structure of the YAML config file
    :param dataset_name: name of the dataset
    :return: modified configuration dict
    """
    if tipo=='LGCN':
        c = 'external.LightGCN_Custom'
    if tipo=='BPRMF':
        c = 'external.BPRMF_Custom'     
    configuration = dict(configuration)
    configuration['experiment']['data_config']['train_path'] = dataset_filepath(dataset_name, 'train')
    configuration['experiment']['data_config']['validation_path'] = dataset_filepath(dataset_name, 'val')
    configuration['experiment']['data_config']['test_path'] = dataset_filepath(dataset_name, 'test')
    configuration['experiment']['dataset'] = dataset_name
    configuration['experiment']['models'][c]['meta']['dataset_name'] = dataset_name
   

        
    return configuration

def set_runtime_metrics_configuration(configuration,dataset_name):
    configuration['experiment']['data_config']['train_path'] = dataset_filepath(dataset_name, 'train')
    configuration['experiment']['data_config']['validation_path'] = dataset_filepath(dataset_name, 'val')
    configuration['experiment']['data_config']['test_path'] = dataset_filepath(dataset_name, 'test')
    configuration['experiment']['models']['RecommendationFolder']['folder'] = get_recs_path(dataset_name)#os.path.abspath('data_dz/' + DATASET + '/' + type)
    configuration['experiment']['dataset'] = dataset_name
    return configuration

def set_model_factors(type : str, configuration: dict,n_factors: int ):
    """Changes the number of fatctors with wich the model will be built with
        :param configuration: dictionary containing the structure of the YAML config file
        :param n_factors: int number of factors
        :return: modified configuration dict
    """
    if type=='LGCN':
        c = 'external.LightGCN_Custom'
    if type=='BPRMF':
        c = 'external.BPRMF_Custom'        
    
    
    configuration['experiment']['models'][c]['factors'] = n_factors
    
    return configuration

def set_model_weights(type : str, configuration: dict,n_factors: int ,DATASET:str):


    if type=='LGCN':
        c = 'external.LightGCN_Custom'
        if DATASET==FACEBOOK:
            configuration['experiment']['models']['external.LightGCN_Custom']['epochs']= 200
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']={}
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['patience'] = 5
            configuration['experiment']['models']['external.LightGCN_Custom']['meta']['validation_rate']= 1
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['mode']= 'auto'
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['monitor']= 'nDCGRendle2020@10'
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['verbose']= True
            
            if(n_factors)==512:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.00011454076953075361
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 1.4623166653305814e-05
                
            if(n_factors)==256:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0001095050831578516
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.009707538210675435
                
            if(n_factors)==128:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.00015011490675299823
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.00025956731556897876
                
            if(n_factors)==64:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0002208073380331132
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.013657214222656406
                
            if(n_factors)==32:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.00012312582826094217
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.09324712179502005
                
            if(n_factors)==16:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0007064551771936087
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.0005499677354860677
                
            if(n_factors)==8:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.00025710837696576766
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.014814373200544878
                
            if(n_factors)==4:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0010026397804933208
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.051207327267699713
                
            if(n_factors)==2:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0007555330659747478
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.020297209644585625
        if DATASET==YAHOO:
            configuration['experiment']['models']['external.LightGCN_Custom']['epochs']= 200
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['patience'] = 5
            configuration['experiment']['models']['external.LightGCN_Custom']['meta']['validation_rate']= 1
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['mode']= 'auto'
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['monitor']= 'nDCGRendle2020@10'
            configuration['experiment']['models']['external.LightGCN_Custom']['early_stopping']['verbose']= True
            if(n_factors)==512:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.00011454076953075361 
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 1.4623166653305814e-05

            if(n_factors)==256:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.000581295268049958 
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.07812719644660297
                
            if(n_factors)==128:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0005348499732957442
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.00029541515935698414
                
            if(n_factors)==64:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0003507478959683115
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.012074840574752454
                
            if(n_factors)==32:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.002049876206747379
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.020075565123230846
                
            if(n_factors)==16:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0018994668577047584
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.08561866884260011
                
            if(n_factors)==8:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0018362366655931806
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 9.788902029962336e-05
                
            if(n_factors)==4:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.0026931211276183566
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.0011566152289430296
                
            if(n_factors)==2:
                configuration['experiment']['models']['external.LightGCN_Custom']['lr']=0.004070364572294006
                configuration['experiment']['models']['external.LightGCN_Custom']['l_w']= 0.011314592969603035
    if type=='BPRMF':
        c = 'external.BPRMF_Custom'
        if DATASET==FACEBOOK:
            if(n_factors)==512:
                configuration['experiment']['models'][c]['lr']=0.00024828109391337706
                configuration['experiment']['models'][c]['l_w']=0.041968799109630556
            if(n_factors)==256:
                configuration['experiment']['models'][c]['lr']=0.000581295268049958
                configuration['experiment']['models'][c]['l_w']=0.07812719644660297
            if(n_factors)==128:
                configuration['experiment']['models'][c]['lr']=0.001985553949455268
                configuration['experiment']['models'][c]['l_w']=0.06445268457836228
            if(n_factors)==64:
                configuration['experiment']['models'][c]['lr']=0.002207733364224528
                configuration['experiment']['models'][c]['l_w']=0.06416374197369942
            if(n_factors)==32:
                configuration['experiment']['models'][c]['lr']=0.00017462489760584107
                configuration['experiment']['models'][c]['l_w']=0.04363346328275595
            if(n_factors)==16:
                configuration['experiment']['models'][c]['lr']=0.0018994668577047584
                configuration['experiment']['models'][c]['l_w']=0.08561866884260011
            if(n_factors)==8:
                configuration['experiment']['models'][c]['lr']=0.00043330573771235717
                configuration['experiment']['models'][c]['l_w']=0.0331391883545839
            if(n_factors)==4:
                configuration['experiment']['models'][c]['lr']=0.0004169080952996497
                configuration['experiment']['models'][c]['l_w']=0.08242471365171306
            if(n_factors)==2:
                configuration['experiment']['models'][c]['lr']=0.0048090994053151944
                configuration['experiment']['models'][c]['l_w']=1.3821952072037158e-05
        
    return configuration

def set_reducers_configuration(type : str, configuration: dict,reducers_types:list,reducers_factors:list,kpca_kernels:list):
    if type=='LGCN':
        c = 'external.LightGCN_Custom'
    if type=='BPRMF':
        c = 'external.BPRMF_Custom'
    
    configuration['experiment']['models'][c]['meta']['reducers_types'] = reducers_types
    configuration['experiment']['models'][c]['meta']['reducers_factors'] = reducers_factors
    configuration['experiment']['models'][c]['meta']['KPCA_Kernels'] =kpca_kernels# ['linear', 'poly','rbf', 'sigmoid','cosine']
    configuration['experiment']['models'][c]['meta']['tsne_reducers_factors'] = [2]
    return configuration

def build_config_file(alg_name):
    configuration = {}
    if alg_name=='LGCN':
        alg = 'external.LightGCN_Custom'
    if alg_name=='LGCN':
        alg =  'external.BPRMF_Custom'
    
    configuration['experiment']={}
    configuration['experiment']['backend']='pytorch'
    configuration['experiment']['data_config']={}
    configuration['experiment']['data_config']['strategy']='fixed'
    configuration['experiment']['data_config']['train_path']=''
    configuration['experiment']['data_config']['validation_path']=''
    configuration['experiment']['data_config']['test_path']=''
    configuration['experiment']['dataset']=''
    configuration['experiment']['top_k']=10
    configuration['experiment']['evaluation']={}
    configuration['experiment']['evaluation']['cutoffs']=[10]
    configuration['experiment']['evaluation']['simple_metrics']=['nDCGRendle2020']
    configuration['experiment']['gpu']=1
    configuration['experiment']['external_models_path']='../external/models/__init__.py'
    configuration['experiment']['models']={}
    configuration['experiment']['models'][alg]={}
    configuration['experiment']['models'][alg]['meta']={}
    configuration['experiment']['models'][alg]['meta']['verbose']: 'true'
    configuration['experiment']['models'][alg]['meta']['save_recs']= 'true'
    configuration['experiment']['models'][alg]['meta']['validation_rate']=''
    configuration['experiment']['models'][alg]['meta']['dataset_name']=''
    configuration['experiment']['models'][alg]['meta']['reducers_types']=''
    configuration['experiment']['models'][alg]['meta']['reducers_factors']=''
    configuration['experiment']['models'][alg]['meta']['KPCA_Kernels']=''
    configuration['experiment']['models'][alg]['meta']['tsne_reducers_factors']=''
    configuration['experiment']['models'][alg]['lr'] = ''
    configuration['experiment']['models'][alg]['epochs'] = ''
    configuration['experiment']['models'][alg]['factors'] = ''
    configuration['experiment']['models'][alg]['batch_size'] = ''
    configuration['experiment']['models'][alg]['l_w'] = ''
    configuration['experiment']['models'][alg]['n_layers'] = ''
    configuration['experiment']['models'][alg]['seed'] = ''
    configuration['experiment']['models'][alg]['early_stopping'] = {}
    configuration['experiment']['models'][alg]['early_stopping']['patience'] =''
    configuration['experiment']['models'][alg]['early_stopping']['mode'] =''
    configuration['experiment']['models'][alg]['early_stopping']['monitor'] =''
    configuration['experiment']['models'][alg]['early_stopping']['verbose'] =''