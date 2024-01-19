from elliot.run import run_experiment
import pickle
import csv
from collections import OrderedDict
import numpy as np

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

#EXPERIMENTAL
def norm_coords(USERS,ITEMS):
    conc =np.concatenate((USERS, ITEMS))
    
    avg_x = sum(conc[:,0]) / len(conc[:,0]) 
    avg_y = sum(conc[:,1]) / len(conc[:,1]) 
    for i in range(len(conc)):
        conc[i][0]-=avg_x
        conc[i][1]-=avg_y
    
    return conc[:len(USERS),:] ,conc[len(USERS):,:]    

