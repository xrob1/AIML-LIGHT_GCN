import compute_processing_data as cp
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np
import torch

#Avvia training e salva componenti per l'analisi
cp.load_data()

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
