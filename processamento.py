from src import *
from src.loader.paths import *
from src.utils.resutls_tab import *
from utils import *

REDUCERS    =['NN','UMAP','KPCA','AUTOE','TSNE','BASE']
FACTORS     =[512,256,128,64,32,16,8,4,2]
EXPLORATION = False

def fai(DATASET,ALGORITMO,FACTORS,REDUCERS,NOME_CSV,TIPO=TEST,EXPLORATION=False):
    # MODELLO PIU GRANDE CON RISPETIVE RIDUZIONI (ALGS)
    config = build_config_file(DATASET, ALGORITMO, FACTORS=FACTORS[0], REDUCERS=REDUCERS, RED_FACTORS=FACTORS[1:],EXPLORATION=EXPLORATION)
    run_on_config(config)
    # MODELLI RIDOTTI  
    for F in FACTORS[1:]:
        config = build_config_file(DATASET, ALGORITMO, FACTORS=F,EXPLORATION=EXPLORATION)
        run_on_config(config)
    
    # ESTRAZIONE E PROCESING RISULTATI CON CSV
    extract_and_process_results(DATASET,ALGORITMO,TIPO)
    # DO CSV
    write_csv(DATASET,OUT_NAME=NOME_CSV)

def fai_exploration(DATASET,ALGORITMO,FACTORS,EXPLORATION=True):
    # MODELLI RIDOTTI  
    for F in FACTORS:
        config = build_config_file(DATASET, ALGORITMO, FACTORS=F,EXPLORATION=EXPLORATION)
        run_on_config(config)

def fai_risultati(DATASET,ALGORITMO,FACTORS,REDUCERS,NOME_CSV,TIPO):
    extract_and_process_results(DATASET,ALGORITMO,TIPO=TIPO)
    write_csv(DATASET,OUT_NAME=NOME_CSV)    
    
#fai(FACEBOOK,LGCN,FACTORS,REDUCERS,'LGCN_FACEBOOK_VALIDATION_EARLY_STOPPIG') 
                 
#fai_exploration(FACEBOOK,BPRMF,FACTORS,REDUCERS,'BPRMF_FACEBOOK',EXPLORATION=True)   
#fai_exploration(YAHOO,BPRMF,FACTORS,REDUCERS,'BPRMF_YAHOO',EXPLORATION=True)                      
#fai_exploration(FACEBOOK,LGCN,FACTORS,REDUCERS,'LGCN_FACEBOOK',EXPLORATION=True)              
#fai_exploration(YAHOO,LGCN,FACTORS,REDUCERS,'LGCN_YAHOO',EXPLORATION=True)   
fai_exploration(MOVIELENS,BPRMF,FACTORS,EXPLORATION=True)   




