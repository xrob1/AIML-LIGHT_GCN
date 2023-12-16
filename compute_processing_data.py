from elliot.run import run_experiment
import pickle

#MOLTO LENTO CON TSNE CON method=exact !!!!!!!!
#Crea i file da leggere con pickle
def load_data():
    run_experiment('config_files/facebook_best_cf_lgcn.yml') #LGCN 
    
#prende 
def get_data():
    file = open(str('models_raw_data/LightGCN_data') , 'rb')
    data = pickle.load(file)
    file.close()
    return data


def get_recs_dict():
    file = open(str('models_raw_data/LightGCN_recs') , 'rb')
    recs = pickle.load(file)
    file.close()
    return recs

