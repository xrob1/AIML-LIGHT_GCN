from src.loader.paths import *
from src import *


def build_config_file( DATASET,ALGORITHM,FACTORS=512,REDUCERS=[],EXPLORATION=False, RED_FACTORS = [256,128,64,32,16,8,4,2] ):
    if ALGORITHM == 'BPRMF':
        ALG = 'external.BPRMF_Custom'
    if ALGORITHM == 'LGCN':
        ALG = 'external.LightGCN_Custom'
    
    if EXPLORATION:
        lr,l_w=get_exp_lrlw(ALGORITHM,DATASET)
    else:
        lr,l_w=get_lrlw(ALGORITHM,DATASET,FACTORS)
    
    configuration = {}

    configuration['experiment']={}
    configuration['experiment']['backend']='pytorch'
    configuration['experiment']['data_config']={}
    configuration['experiment']['data_config']['strategy']='fixed'
    configuration['experiment']['data_config']['train_path']=       dataset_filepath(DATASET, 'train')
    configuration['experiment']['data_config']['validation_path']=  dataset_filepath(DATASET, 'val')
    configuration['experiment']['data_config']['test_path']=        dataset_filepath(DATASET, 'test')
    configuration['experiment']['dataset']= DATASET
    configuration['experiment']['top_k']=10
    configuration['experiment']['evaluation']={}
    configuration['experiment']['evaluation']['cutoffs']=[10]
    configuration['experiment']['evaluation']['simple_metrics']=['nDCGRendle2020']
    configuration['experiment']['gpu']=1
    configuration['experiment']['external_models_path']='../external/models/__init__.py'
    
    configuration['experiment']['models']={}    
    configuration['experiment']['models'][ALG]={}
    
    configuration['experiment']['models'][ALG]['meta']={}
    configuration['experiment']['models'][ALG]['meta']['verbose']= 'true'
    configuration['experiment']['models'][ALG]['meta']['save_recs']= 'true'
    configuration['experiment']['models'][ALG]['meta']['validation_rate']=  5
    configuration['experiment']['models'][ALG]['meta']['dataset_name']= DATASET
    configuration['experiment']['models'][ALG]['meta']['reducers_types']=REDUCERS
    configuration['experiment']['models'][ALG]['meta']['reducers_factors']=RED_FACTORS
    configuration['experiment']['models'][ALG]['meta']['KPCA_Kernels']= ['linear', 'poly','rbf', 'sigmoid','cosine']
    
    
    if EXPLORATION : 
        configuration['experiment']['models'][ALG]['meta']['hyper_max_evals']= 20
        configuration['experiment']['models'][ALG]['meta']['hyper_opt_alg']= 'tpe'
        configuration['experiment']['models'][ALG]['meta']['validation_metric']= 'nDCGRendle2020@10'
        configuration['experiment']['models'][ALG]['meta']['validation_rate']=  1
        
    
    configuration['experiment']['models'][ALG]['epochs'] = 200
    configuration['experiment']['models'][ALG]['factors'] = FACTORS
    
    if DATASET==MOVIELENS:
        configuration['experiment']['models'][ALG]['batch_size'] = 2048
    else:
        configuration['experiment']['models'][ALG]['batch_size'] = 64
        
    configuration['experiment']['models'][ALG]['l_w'] = l_w
    configuration['experiment']['models'][ALG]['lr'] =  lr
    
    if ALGORITHM=='LGCN' : configuration['experiment']['models'][ALG]['n_layers'] = 3
    
    configuration['experiment']['models'][ALG]['seed'] = 123
    if EXPLORATION:
        configuration['experiment']['models'][ALG]['early_stopping'] = {}
        configuration['experiment']['models'][ALG]['early_stopping']['patience'] =5
        configuration['experiment']['models'][ALG]['early_stopping']['mode'] ='auto'
        configuration['experiment']['models'][ALG]['early_stopping']['monitor'] ='nDCGRendle2020'
        configuration['experiment']['models'][ALG]['early_stopping']['verbose'] ='true'
    
    return configuration

def get_lrlw(ALG,DATASET,FACTORS):

    if ALG=='LGCN':
        if DATASET == FACEBOOK:
            if FACTORS == 512: lr =0.00011454076953075361; lw= 1.4623166653305814e-05
            if FACTORS == 256: lr =0.0001095050831578516; lw= 0.009707538210675435
            if FACTORS == 128: lr =0.00015011490675299823; lw= 0.00025956731556897876
            if FACTORS == 64: lr =0.0002208073380331132; lw= 0.013657214222656406
            if FACTORS == 32: lr =0.00012312582826094217; lw= 0.09324712179502005
            if FACTORS == 16: lr =0.0007064551771936087; lw= 0.0005499677354860677
            if FACTORS == 8: lr =0.00025710837696576766; lw= 0.014814373200544878
            if FACTORS == 4: lr =0.0010026397804933208; lw= 0.051207327267699713
            if FACTORS == 2: lr =0.0007555330659747478; lw= 0.020297209644585625
        if DATASET == YAHOO:
            if FACTORS == 512: lr =0.00011454076953075361 ;lw= 1.4623166653305814e-05
            if FACTORS == 256:lr =0.000581295268049958 ; lw= 0.07812719644660297
            if FACTORS == 128:lr =0.0005348499732957442; lw= 0.00029541515935698414
            if FACTORS == 64:lr =0.0003507478959683115;lw= 0.012074840574752454
            if FACTORS == 32:lr =0.002049876206747379;lw= 0.020075565123230846
            if FACTORS == 16:lr =0.0018994668577047584;lw= 0.08561866884260011
            if FACTORS == 8:lr =0.0018362366655931806;lw= 9.788902029962336e-05
            if FACTORS == 4:lr =0.0026931211276183566;lw= 0.0011566152289430296
            if FACTORS == 2:lr =0.004070364572294006;lw= 0.011314592969603035
        """
        if DATASET == MOVIELENS:
            if FACTORS == 512:
            if FACTORS == 256:
            if FACTORS == 128:
            if FACTORS == 64:
            if FACTORS == 32:
            if FACTORS == 16:
            if FACTORS == 8:
            if FACTORS == 4:
            if FACTORS == 2:
        """
    if ALG=='BPRMF':
        if DATASET == FACEBOOK:
            if FACTORS == 512: lr =0.00024828109391337706;lw =0.041968799109630556
            if FACTORS == 256: lr =0.000581295268049958;lw =0.07812719644660297
            if FACTORS == 128: lr =0.001985553949455268;lw =0.06445268457836228
            if FACTORS == 64: lr =0.002207733364224528;lw =0.06416374197369942
            if FACTORS == 32: lr =0.00017462489760584107;lw =0.04363346328275595
            if FACTORS == 16: lr =0.0018994668577047584;lw =0.08561866884260011
            if FACTORS == 8: lr =0.00043330573771235717;lw =0.0331391883545839
            if FACTORS == 4: lr =0.0004169080952996497;lw =0.08242471365171306
            if FACTORS == 2: lr =0.0048090994053151944;lw =1.3821952072037158e-05
        if DATASET == YAHOO:
            if FACTORS == 512: lr=0.0005754258527936548;lw=0.007115052124575682
            if FACTORS == 256: lr=0.00046083949100544284;lw=0.0022183898072497647	
            if FACTORS == 128:lr=0.0005880195270903159;lw=0.002361872541284341
            if FACTORS == 64:lr=0.0024523481154271644;lw=0.024769137059383623	
            if FACTORS == 32:lr=0.002049876206747379;lw=0.020075565123230846	
            if FACTORS == 16:lr=0.0018994668577047584;lw=0.08561866884260011
            if FACTORS == 8: lr=0.004250078801463418;lw=0.05993472872034193
            if FACTORS == 4:lr=0.0010026397804933208;lw=0.051207327267699713	
            if FACTORS == 2:lr=0.00037151956891814914;lw=0.05675791581213648
        
    return lr,lw
def get_exp_lrlw(ALG,DATASET):
    if DATASET==FACEBOOK:
        if ALG=='LGCN':return   [ 'loguniform', -9.210340372, -5.298317367 ],[' loguniform', -11.512925465, -2.30258509299 ]
        if ALG=='BPRMF':return  [ 'loguniform', -9.210340372, -5.298317367 ],[ 'loguniform', -11.512925465, -2.30258509299 ]
    if DATASET==YAHOO:
        if ALG=='LGCN': return  [ 'loguniform', -9.210340372, -5.298317367 ],[ 'loguniform', -11.512925465, -2.30258509299 ]
        if ALG=='BPRMF':return  [ 'loguniform', -9.210340372, -5.298317367 ],[ 'loguniform', -11.512925465, -2.30258509299 ]
    if DATASET==MOVIELENS:
        if ALG=='LGCN':return   [ 'loguniform', -9.210340372, -5.298317367 ],[ 'loguniform', -11.512925465, -2.30258509299 ]
        if ALG=='BPRMF':return  [ 'loguniform', -9.210340372, -5.298317367 ],[ 'loguniform', -11.512925465, -2.30258509299 ]
        

def build_runtime_config_file(DATASET):
    configuration = {}
    
    configuration['experiment']={}
    configuration['experiment']['backend']= 'pytorch'
    configuration['experiment']['data_config']={}
    configuration['experiment']['data_config']['strategy']= 'fixed'
    configuration['experiment']['data_config']['train_path']= dataset_filepath(DATASET, 'train')
    configuration['experiment']['data_config']['validation_path']=dataset_filepath(DATASET, 'val')
    configuration['experiment']['data_config']['test_path']= dataset_filepath(DATASET, 'test')
    configuration['experiment']['dataset']= DATASET
    configuration['experiment']['top_k']= 10
    configuration['experiment']['evaluation']={}
    configuration['experiment']['evaluation']['cutoffs']= [10]
    configuration['experiment']['evaluation']['paired_ttest']= 'false'
    configuration['experiment']['evaluation']['simple_metrics']= ['nDCGRendle2020', 'HR', 'Precision', 'Recall']
    configuration['experiment']['gpu']= 0
    configuration['experiment']['models']={}
    configuration['experiment']['models']['RecommendationFolder']={}
    configuration['experiment']['models']['RecommendationFolder']['folder']=  RECS_DIRECTORY
    
    return configuration

