from elliot.run import run_experiment
import pickle

#Crea i file da leggere con pickle
def load_data():
    run_experiment('config_files/facebook_best_cf_lgcn.yml') #LGCN 
    
#prende 
def get_data_container():
    file = open(str('models_raw_data/LightGCN_data') , 'rb')
    data = pickle.load(file)
    file.close()
    return data[0]

def get_data(i):
    file = open(str('models_raw_data/LightGCN_recommendations') , 'rb')
    rec = pickle.load(file)
    file.close()
    return rec[0][i]

def get_val_recommandation():   
    return get_data(0) 

def get_test_recommandation():
    return get_data(1) 

def get_user_embedding():
    return get_data(2) 

def get_item_embedding():
    return get_data(3) 

def get_offsets():
    return get_data(4) 

def get_predictions():
    return get_data(5) 

def get_public_items():
    data = get_data_container()
    return data.public_items