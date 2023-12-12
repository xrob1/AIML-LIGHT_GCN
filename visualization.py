from elliot.run import run_experiment
import pickle
import matplotlib.pyplot as plt

def load_data(name='LightGCN_data'):
    file = open(str('models_raw_data/'+name) , 'rb')
    data = pickle.load(file)
    file.close()
    return data[0]

def load_recommandations(name='LightGCN_recommendations'):
    file = open(str('models_raw_data/'+name) , 'rb')
    rec = pickle.load(file)
    file.close()
    return rec[0][0],rec[0][1],rec[0][2],rec[0][3],rec[0][4],rec[0][5]

#Da migliorare (too slow?)
def get_i_public_train_dict(data):
    public_i_train_dict={}
    for key in data.i_train_dict:
        public_i_train_dict[data.private_users[key]]=[]
        for item in list(data.i_train_dict[key].keys()):
            public_i_train_dict[data.private_users[key]].append(data.private_items[item])

    return  public_i_train_dict

def train():
    run_experiment('config_files/facebook_best_cf_lgcn.yml') #LGCN 

#train()
data = load_data()

test_dict = data.test_dict
vald_dict = data.val_dict
recm_val,recm_test,gu,gi,offsets,preds = load_recommandations()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go





#ITEMS
items=gi.cpu().detach().numpy()
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(items)
fig1 = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1],  color_discrete_sequence=['blue'])
fig1.show()
#USER
users=gu.cpu().detach().numpy()
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(users)
fig2 = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1],  color_discrete_sequence=['blue'])

#fig3 = go.Figure(data=fig1.data + fig2.data)
fig2.show()






