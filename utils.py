import ruamel.yaml as YAML
import pickle
from elliot.run import run_experiment
from src import *
from src.loader.paths import *
from src.utils.resutls_tab import *
import glob 
import shutil
import csv

#PICKLE METHOD TO OPEN RAW DATA
def get_rdata(DATASET,F_NAME,ALG):
    with open( raw_file_path(DATASET,F_NAME,ALG), 'rb') as file:
        recs = pickle.load(file)
    return recs

#SAVE CONFIGURATION AND START EXPERIMENT
def run_on_config(conf,conf_path = basic_conf_file() ):
    save_config(conf,conf_path)
    start(conf_path)

#SAVE CONFIGURATION     
def save_config(conf,conf_path = basic_conf_file()):
    yaml = YAML.YAML()
    with open(conf_path, 'w') as file:
        yaml.dump(conf, file)

#START EXPERIMENT
def start(conf_path = basic_conf_file() ):
    run_experiment(conf_path) 

#CLEAR TMP RECS DIR
def clean_tmp_rec_dir():
    files = glob.glob(RECS_DIRECTORY+'*')  
    for f in files:
        os.remove(f)

#CLEAR ELIOT RESULT DIRECTORY
def clear_results_dir(DATASET):
    files = glob.glob(dataset_results_path(DATASET,'weights')+'/*') 
    for f in files:
        shutil.rmtree(f)
    files = glob.glob(dataset_results_path(DATASET,'performance')+'/*') 
    for f in files:
        os.remove(f) 
    files = glob.glob(dataset_results_path(DATASET,'recs')+'/*') 
    for f in files:
        os.remove(f) 

   
#EXTRACT RECS FROM RAW FILES
def extract_recs(DATASET,ALGORITMO,TIPO='test',CLEAN=True):
    
    if CLEAN:
        clean_tmp_rec_dir()
    
    files = raw_files_names_list(DATASET,ALGORITMO)    
    for F_NAME in files:         
        RECS = get_rdata(DATASET,F_NAME,ALGORITMO)[TIPO]    
        METHOD = F_NAME.split("_")[2]     
        if(METHOD=='KPCA'):
            for KERNEL in RECS.keys():
                save_tsv(RECS[KERNEL],str(METHOD+'_'+KERNEL))
        else:
            save_tsv(RECS,METHOD)

#SAVES RECS TO TEMP REC DIR IN TSV FORMAT
def save_tsv(RECS,METHOD):
    SIZES =  RECS.keys()
    for SIZE in SIZES:
        REC_NAME =str(METHOD+'@'+str(SIZE)+'.tsv')
        SAVE_PATH = os.path.join(RECS_DIRECTORY,REC_NAME)  
        with open(SAVE_PATH,'w') as file:
            for USER in RECS[SIZE]:
                for LINE in RECS[SIZE][USER]:
                    file.write(str(str(USER) + "	" + str(LINE[0]) + "	" + str(LINE[1]) + '\n'))

#ESTRAE e avvia il processing dei risultati dato un dataset e un metodo di raccomandazione
def extract_and_process_results(DATASET,ALGORITMO):
    extract_recs(DATASET,ALGORITMO,TEST)
    clear_results_dir(DATASET)
    run_on_config(build_runtime_config_file(DATASET))

#CREATE CSV FILE FROM RESULTS
def write_csv(DATASET,OUT_NAME='results_'):
    results = []
    for f in [f for f in os.listdir(dataset_results_path(DATASET))]:
        if f.split('_')[0] != 'rec': continue
        with open(os.path.join(dataset_results_path(DATASET),f) , 'r') as file:
            fields = file.readline()  # SALTA INTESTAZIONE:
            for line in file:
                line = line.replace('.',',')
                line=line.split('\t')
                method = line[0].split('@')[0]
                n = line[0].split('@')[1]
                results.append([int(n), method, line[1][:6], line[2][:6], line[3][:6], line[4][:6]])
    results = sorted(results, reverse=True)
    with open(OUT_NAME + '.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n')
        csvwriter.writerow(['|e|', 'Metodo', 'nDCGRendle2020', 'HR', 'Precision', 'Recall'])
        csvwriter.writerows(results)