from elliot.run import run_experiment
import pickle

#MOLTO LENTO CON TSNE CON method=exact !!!!!!!!
#Crea i file da leggere con pickle
def load_data(config_file = 'custom_configs/facebook_best_cf_lgcn.yml'):
    run_experiment(config_file) #LGCN 
    
#prende 
def get_data():
    file = open(str('models_raw_data/LightGCN_data') , 'rb')
    data = pickle.load(file)
    file.close()
    return data

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

def save_recs(name,recs):
    for SIZE in recs:		
        path = 'data_dz/facebook/'+name+'/'+name+'@'+str(SIZE)+'.tsv'        
        file = open(path, 'w')		
        for USER in recs[SIZE]:
            for rec in recs[SIZE][USER]:
                file.write(str(str(USER)+"	"+str(rec[0])+"	"+str(rec[1])+'\n'))
        file.close()   

def save_recs_at(name,recs,SIZE):	
    path = 'data_dz/facebook/'+name+'/'+name+'@'+str(SIZE)+'.tsv'        
    file = open(path, 'w')		
    for USER in recs[SIZE]:
        for rec in recs[SIZE][USER]:
            file.write(str(str(USER)+"	"+str(rec[0])+"	"+str(rec[1])+'\n'))
    file.close()   