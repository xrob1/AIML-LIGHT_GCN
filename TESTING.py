import compute_processing_data as cp
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np
import torch
from elliot.recommender.generic.Proxy import ProxyRecommender
from elliot.run import run_experiment



#Avvia training e salva componenti per l'analisi
cp.load_data()

#Prende dizionario con tutte le raccomandazioni fra i vari metodi
recs = cp.get_recs_dict()

#Salvataggio Raccomandazioni per validation_set per ogni modalità di predizione

DATASET = "facebook/"
MODE = "validation"

for method in recs:	
	for size in recs[method][MODE]:		
		file = open('data_dz/facebook/'+str(method)+'_@'+str(size)+'.tsv', 'w')		
		for user in recs[method][MODE][size]:
			for rec in recs[method][MODE][size][user]:
				file.write(str(str(user)+"	"+str(rec[0])+"	"+str(rec[1])+'\n'))
		file.close()   

run_experiment(f"config_files/runtime_metrics_conf.yml")
"""
print("OK?!?")




#Caricamento componenti utili
public_items=cp.get_public_items()          #Lista traduzione publici a privati
val_rec = cp.get_val_recommandation()       #RACOMANDAZIONI VALIDATION
items = cp.get_item_embedding().cpu().detach().numpy()  #Embedding Oggetti
users = cp.get_user_embedding().cpu().detach().numpy()  #Embedding Utenti

#Estrazione indici oggetti per ogni utente
i_val_rec=[]                                #Per ogni utente indici oggetti raccomandati
for key in val_rec: 
	i_val_rec.append([e[0] for e in val_rec[key]])


#TNSE Della Concatenazione fra oggetti e utenti
concatenation = np.concatenate((items,users)) #Concatenazione 
tsne = TSNE(n_components=2, random_state=42)

i_u_concat = tsne.fit_transform(concatenation)
i_tsne =    i_u_concat[:len(items),:]
u_tsne =    i_u_concat[len(items):,:]

#visualizzazione
fig_obj  = px.scatter(x=i_tsne[:, 0], y=i_tsne[:, 1],  color_discrete_sequence=['blue'])
fig_usr  = px.scatter(x=u_tsne[0:1, 0], y=u_tsne[0:1, 1],  color_discrete_sequence=['red'])#1 user
#fig3 = go.Figure(data=fig1.data + fig2.data)
#fig3.show()

max([public_items[e] for e in [key for key in public_items.keys()]])    # indice privato massimo
max(public_items.keys())                                                # indice pubblico massimo  
max([e[0] for key in val_rec for e in val_rec[key]])                    # indice item massimo in raccomandazione
len(items)                                                              # numero oggetti
len(i_tsne)                                                             # numero oggetti Itnse

len(users)
len(u_tsne)
len(val_rec)

test_batch=[]
for e in i_val_rec[0]:   
	test_batch.append(i_tsne[public_items[e]])
test_batch=np.array(test_batch)

fig_test  = px.scatter(x=test_batch[:, 0], y=test_batch[:, 1],  color_discrete_sequence=['black'])
fig_usr  = px.scatter(x=u_tsne[:, 0], y=u_tsne[:, 1],  color_discrete_sequence=['red'])#1 user
fig_test_usr = go.Figure(data= fig_usr.data +fig_test.data )
fig_test_usr.show()


if(VISUALIZATION):
    users,items =cp.get_umap_data()
    usr1_recs = []
    p_it=cp.get_data().public_items
    
    itms=[p_it[e] for e in [50,19,3,173,305,53,163,77,250,288]]
    for e in itms:
        usr1_recs.append(items[e])
    usr1_recs=np.array(usr1_recs)

    fig_itm  = px.scatter(x=items[:, 0], y=items[:, 1],  color_discrete_sequence=['green'])    
    fig_rec  = px.scatter(x=usr1_recs[:, 0], y=usr1_recs[:, 1],  color_discrete_sequence=['blue'])
    fig_usr  = px.scatter(x=users[:, 0], y=users[:, 1],  color_discrete_sequence=['red'])#1 user
    fig3 = go.Figure(data=fig_itm.data + fig_rec.data)
    fig3 = go.Figure(data=fig3.data + fig_usr.data)
    fig3.show()


"""