from elliot.run import run_experiment
import pickle
import csv
from collections import OrderedDict
import numpy as np
from src import *
from src.loader.paths import *
from src.utils.resutls_tab import *
import glob

#MOLTO LENTO CON TSNE CON method=exact !!!!!!!!
#Crea i file da leggere con pickle
def load_data(config_file = 'custom_configs/facebook_best_cf_lgcn.yml'):
    run_experiment(config_file) #LGCN 
    
#prende 
def get_data():
    file = open(str('models_raw_data/LightGCN_data') , 'rb')
    data = pickle.load(file)
    file.close()
    return data[0]

def get_base_recs():
    file = open(str('models_raw_data/LightGCN_base_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs
def get_pca_recs():
    file = open(str('models_raw_data/LightGCN_PCA_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs
def get_kpca_recs():
    file = open(str('models_raw_data/LightGCN_KPCA_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs
    
def get_tsne_recs():
    file = open(str('models_raw_data/LightGCN_tsne_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs

def get_lle_recs():
    file = open(str('models_raw_data/LightGCN_lle_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs

def get_isomap_recs():
    file = open(str('models_raw_data/LightGCN_isomap_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs

def get_umap_recs():
    file = open(str('models_raw_data/LightGCN_umap_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs

def get_umap_data():
    file = open(str('models_raw_data/LightGCN_umap_data') , 'rb')
    data = pickle.load(file)
    file.close()
    return data[0].cpu().detach().numpy(),data[1].cpu().detach().numpy() #USERS ITEMS

def get_recs_from_file(path , conversion = True):
    public_items = get_data().public_items
    public_users = get_data().public_users
    recs={}
    with open(path) as file:        
        tsv_file = csv.reader(file, delimiter="\t")        
        previous=-1
        for line in tsv_file:
            if conversion:
                try: id,item,score = public_users[int(line[0])],public_items[int(line[1])],float(line[2]) #OGGETTI NON PRESENTI?????????
                except:continue
            else:
                id,item,score = line[0],line[1],line[2]
            
            if(id!=previous or previous==-1):
                previous=id
                recs[id]=[]
            recs[id].append(item)
        
    return recs


def get_user_recs(method='base' , dimension='64',conversion=True):
    path = 'data_dz/facebook/'+method+'/'+method+'@'+dimension+'.tsv'
    return get_recs_from_file(path,conversion)

def get_user_test_recs(conversion=True):
    return get_recs_from_file('data/facebook_book/test.tsv',conversion)


def get_user_train_recs(conversion=True):
    return get_recs_from_file('data/facebook_book/train.tsv',conversion)


def get_hot_users(min_ratings=5):
    public_items = get_data().public_items
    public_users = get_data().public_users
    recs={}
    ratings=[]
    with open(str('data/facebook_book/train.tsv')) as file:        
        tsv_file = csv.reader(file, delimiter="\t")        
        previous=-1
        for line in tsv_file:
            try: id,item,score = public_users[int(line[0])],public_items[int(line[1])],float(line[2]) #OGGETTI NON PRESENTI?????????
            except:continue
            
            if(id!=previous or previous==-1):
                
                
                if len(ratings)>=min_ratings:
                    recs[previous]=ratings                 
                    
                previous=id
                ratings=[]                
            
            ratings.append(item)
   
    recs = {k: v for k, v in sorted(recs.items(), key=lambda item: len(item[1]))}
    recs = OrderedDict(reversed(list(recs.items())))
 
    return recs

def get_cold_users(min_ratings=4):
    public_items = get_data().public_items
    public_users = get_data().public_users
    recs={}
    ratings=[]
    with open(str('data/facebook_book/train.tsv')) as file:        
        tsv_file = csv.reader(file, delimiter="\t")        
        previous=-1
        for line in tsv_file:
            try: id,item,score = public_users[int(line[0])],public_items[int(line[1])],float(line[2]) #OGGETTI NON PRESENTI?????????
            except:continue
            
            if(id!=previous or previous==-1):               
                
                if len(ratings)<=min_ratings and previous!=-1:
                    recs[previous]=ratings                    
                    
                previous=id
                ratings=[]                
            
            ratings.append(item)
    recs = {k: v for k, v in sorted(recs.items(), key=lambda item: len(item[1]))}
    

    return recs

def best_users(path,treshold=0.5):
    stats = hr_on_users(path)
    stats_cut={}

    for e in stats:
        if stats[e][1]>=treshold:
            stats_cut[e]=stats[e]
    stats=stats_cut ; del stats_cut
    
    stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1][1])}
    
    stats = OrderedDict(reversed(list(stats.items())))
    return(stats)

def worst_users(path,treshold=0.5):
    stats = hr_on_users(path)
    stats_cut={}

    for e in stats:
        if stats[e][1]<=treshold:
            stats_cut[e]=stats[e]
    stats=stats_cut ; del stats_cut
    
    stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1][1])}

    return(stats)
    
    
    
    
    
    
    
    
def hr_on_users(path , test='data/facebook_book/test.tsv' ):
    stats = {}
 
    TEST = get_recs_from_file(test,conversion=True)
    RECS = get_recs_from_file(path,conversion=True)
    
    for user in RECS.keys():
        HITS = len( set(TEST[user]) & set(RECS[user]) ) 
        HR = HITS/len(TEST[user])
        stats[user] = [HITS , HR]

    return(stats)
 
def get_rdata(DATASET,F_NAME,ALG):
    with open( raw_file_path(DATASET,F_NAME,ALG), 'rb') as file:
        recs = pickle.load(file)
    return recs

def save_recs(ALG,F_NAME, DATASET,type='validation',KPCA=False):       
    RECS = get_rdata(DATASET,F_NAME,ALG)[type]    
    METHOD = F_NAME.split("_")[2]
     
    if(KPCA):
        for KERNEL in RECS.keys():
            save_tsv(DATASET,RECS[KERNEL],str(METHOD+'_'+KERNEL),ALG)
    else:
        save_tsv(DATASET,RECS,METHOD,ALG)
  

def save_tsv(DATASET,RECS,METHOD,ALG):
    SIZES =  RECS.keys()
    for SIZE in SIZES:
        REC_NAME =str(METHOD+'@'+str(SIZE)+'.tsv')
        SAVE_PATH = os.path.join(get_recs_path(DATASET,ALG),REC_NAME)  
        with open(SAVE_PATH,'w') as file:
            for USER in RECS[SIZE]:
                for LINE in RECS[SIZE][USER]:
                    file.write(str(str(USER) + "	" + str(LINE[0]) + "	" + str(LINE[1]) + '\n'))

def clear_results_dir(DATASET):
    files = glob.glob(dataset_results_path(DATASET)+'/*')  # [f for f in os.listdir('results/facebook_book/performance/')]
    for f in files:
        os.remove(f)

import ruamel.yaml
import ruamel.yaml.util
yaml = ruamel.yaml.YAML()
def save_config(CONFIG_FILE,configuration):
    with open(CONFIG_FILE, 'w') as file:
        yaml.dump(configuration, file)

def open_config(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as file:
        configuration = yaml.load(file)
    return configuration

def write_results_csv(DATASET,out_f_name='results_'):


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

    with open(out_f_name + DATASET + '.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n')
        csvwriter.writerow(['|e|', 'Metodo', 'nDCGRendle2020', 'HR', 'Precision', 'Recall'])
        csvwriter.writerows(results)

