from src import *
from src.loader.paths import *
from src.utils.resutls_tab import *
from utils import *

REDUCERS    =['NN','UMAP','KPCA','AUTOE','TSNE','BASE']
FACTORS     =[512,256,128,64,32,16,8,4,2]
EXPLORATION = False

def fai(DATASET,ALGORITMO,FACTORS,REDUCERS,NOME_CSV):
    # MODELLO PIU GRANDE CON RISPETIVE RIDUZIONI (ALGS)
    config = build_config_file(DATASET, ALGORITMO, FACTORS=FACTORS[0], REDUCERS=REDUCERS, RED_FACTORS=FACTORS[1:])
    run_on_config(config)
    # MODELLI RIDOTTI  
    for F in FACTORS[1:]:
        config = build_config_file(DATASET, ALGORITMO, FACTORS=F)
        run_on_config(config)
    # ESTRAZIONE E PROCESING RISULTATI CON CSV
    extract_and_process_results(DATASET,ALGORITMO)
    # DO CSV
    write_csv(DATASET,OUT_NAME=NOME_CSV)



                 
fai(FACEBOOK,BPRMF,FACTORS,REDUCERS,'BPRMF_FACEBOOK')
fai(YAHOO,BPRMF,FACTORS,REDUCERS,'BPRMF_YAHOO')                    
fai(FACEBOOK,LGCN,FACTORS,REDUCERS,'LGCN_FACEBOOK')              
fai(YAHOO,LGCN,FACTORS,REDUCERS,'LGCN_YAHOO')




