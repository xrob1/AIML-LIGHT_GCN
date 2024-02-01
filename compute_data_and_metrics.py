import compute_processing_data as cp
from elliot.run import run_experiment
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from elliot import user_stats as us

STARTUP=True
EXPERIMENT=False
VISUALIZATION=False

if(STARTUP):
    #Avvia training e salva componenti per l'analisi
    cp.load_data('config_files/custom.yml')
    
    #Salvataggio Raccomandazioni validation_set per ogni modalità
    cp.save_recs('base',cp.get_base_recs()['validation'])
    cp.save_recs_at('umap',cp.get_umap_recs()['validation'],2)

if(EXPERIMENT):
    #Avvio calcolo metriche, 
    run_experiment(f"custom_configs/umap_on_validation.yml")
    run_experiment(f"custom_configs/base_on_validation.yml")

if(VISUALIZATION):    
    import matplotlib.pyplot as plt
    
    #OVERFIT = 
    
    HOT_USERS   =   cp.get_hot_users(10)
    COLD_USERS  =   cp.get_cold_users(3)
    BEST_PERFORMANCE_USERS  = cp.best_users('data_dz/facebook/base/base@64.tsv')
    WORST_PERFORMANCE_USERS = cp.worst_users('data_dz/facebook/base/base@64.tsv')
    HR = cp.hr_on_users('data_dz/facebook/base/base@64.tsv')
    
    #get coordinate oggetti da visualizzare umap
    users,items =cp.get_umap_data()
    
    #EXPERIMENTAL
    #users,items = cp.norm_coords(users,items)
    
    fig = plt.figure()
    fig2 = plt.figure()
    
    #UTENTE ITEM RACCOMANDATI
    recs_lgcn=          cp.get_user_recs()              #BASE   ids
    recs_lgcn_ridotto=  cp.get_user_recs('umap','2')    #UMAP@2 ids
    
    
    #ID UTENTE DA VISUALIZZARE
    ID_USER  =   [e for e in recs_lgcn.keys()][4] #LGCN
    ID_USER  =   [e for e in BEST_PERFORMANCE_USERS.keys()][0] #BEST USER
    ID_USER  =   [e for e in WORST_PERFORMANCE_USERS.keys()][0] #WORST USER
    #ID_USER  =   [e for e in HOT_USERS.keys()][0] #HOTTEST USER
    #ID_USER  =   [e for e in COLD_USERS.keys()][0] #COLDEST USER
    

    #COORDINATE OGGETTI DA VISUALIZZARE
    ITEMS_lgcn  =   np.array([items[e] for e in recs_lgcn[ID_USER]])
    ITEMS_lgcn_ridotto  =   np.array([items[e] for e in recs_lgcn_ridotto[ID_USER]])
    
    #COORDINATE UTENTE DA VISUALIZARE
    USER_lgcn   =   np.array(users[ID_USER])
    USER_lgcn_ridotto   =   np.array(users[ID_USER])
    
    #PLOT
    ax1 = fig.add_subplot(221)  #BASE
    ax1.set_title("BASE ")
    ax1.scatter(ITEMS_lgcn[:,0], ITEMS_lgcn[:,1] ,  s=10, c='y', marker="h", label='ITEMS RACCOMANDATI')
    ax1.scatter(USER_lgcn[0], USER_lgcn[1] ,    s=10, c='r', marker="s", label='UTENTE')
    
    ax2 = fig.add_subplot(222)  #LGCN RIDOTTO
    ax2.set_title("UMAP ")
    ax2.scatter(ITEMS_lgcn_ridotto[:,0], ITEMS_lgcn_ridotto[:,1] ,  s=10, c='y', marker="h")#, label='ITEMS RACCOMANDATI')
    ax2.scatter(USER_lgcn_ridotto[0], USER_lgcn_ridotto[1] ,    s=10, c='r', marker="s")#, label='UTENTE')
    
    
    
    
    #UTENTE ITEM NEL TEST   (DA RACCOMANDARE)
    TEST_recs = cp.get_user_test_recs()
    TEST_ITEMS=np.array([items[e] for e in TEST_recs[ID_USER]])
    
    ax3 = fig.add_subplot(223)  #TEST RECS
    ax3.set_title("TO RECOMMAND")
    ax3.scatter(TEST_ITEMS[:,0], TEST_ITEMS[:,1]            ,  s=10, c='b', marker="p", label='DA RACCOMANDARE')
    ax3.scatter(USER_lgcn_ridotto[0], USER_lgcn_ridotto[1]  ,  s=10, c='r', marker="s")
    

    
    #UETNTE ITEM NEL TRAIN  (INFO DI PARTENZA)
    TRAIN_recs = cp.get_user_train_recs()
    TRAIN_ITEMS=np.array([items[e] for e in TRAIN_recs[ID_USER]])
    
    ax4 = fig.add_subplot(224)  #TEST RECS
    ax4.set_title("TRAIN SET ")
    ax4.scatter(TRAIN_ITEMS[:,0], TRAIN_ITEMS[:,1]            ,  s=10, c='g', marker="o", label='ITEM TRAIN')
    ax4.scatter(USER_lgcn_ridotto[0], USER_lgcn_ridotto[1]  ,  s=10, c='r', marker="s")
   
    
    fig.legend()
    fig.show()
    
    
    #PLOT INSIEME
    ax5 = fig2.add_subplot()
    ax5.set_title("HR: "+str(HR[ID_USER]))
    ax5.scatter(ITEMS_lgcn[:,0], ITEMS_lgcn[:,1] ,                  s=10, c='y', marker="h", label='ITEMS RACCOMANDATI BASE')
    #ax5.scatter(ITEMS_lgcn_ridotto[:,0], ITEMS_lgcn_ridotto[:,1] ,  s=10, c='m', marker="v",label='ITEMS RACCOMANDATI (R)')
    ax5.scatter(USER_lgcn[0], USER_lgcn[1] ,                        s=10, c='r', marker="s", label='UTENTE')
    ax5.scatter(TEST_ITEMS[:,0], TEST_ITEMS[:,1]            ,       s=10, c='b', marker="p", label='DA RACCOMANDARE')
    ax5.scatter(TRAIN_ITEMS[:,0], TRAIN_ITEMS[:,1]            ,     s=10, c='g', marker="o", label='ITEM TRAIN')
    fig2.legend()
    fig2.show()
    
    plt.show(block=True)
    

    
    
    print("END")
        

    
    """
    usr_recs_coord=np.array([items[e] for e in recs[USER]])
    usr_test_recs_coord=np.array([items[e] for e in user_test_recs[USER]])
    usr_coords=np.array(users[USER])
    usr_train_recs_coord=np.array([items[e] for e in user_train_recs[USER]])
    
    
    
    fig = plt.figure()
    
    
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    

    ax1.scatter(usr_train_recs_coord[:,0], usr_train_recs_coord[:,1] ,              s=10, c='y', marker="h", label='items_train')
    ax4.scatter(usr_train_recs_coord[:,0], usr_train_recs_coord[:,1] ,              s=10, c='y', marker="h", label='items_train')
    
    ax2.scatter(usr_recs_coord[:,0], usr_recs_coord[:,1] ,              s=10, c='b', marker="s", label='items_recommanded')
    ax4.scatter(usr_recs_coord[:,0], usr_recs_coord[:,1] ,              s=10, c='b', marker="s", label='items_recommanded')
    
    ax3.scatter(usr_test_recs_coord[:,0], usr_test_recs_coord[:,1] ,    s=10, c='g', marker='p', label='items_test')
    ax4.scatter(usr_test_recs_coord[:,0], usr_test_recs_coord[:,1] ,    s=10, c='g', marker='p', label='items_test')
    
    ax1.scatter(usr_coords[0],usr_coords[1],                            s=10, c='r', marker="o", label='user')
    ax2.scatter(usr_coords[0],usr_coords[1],                            s=10, c='r', marker="o", label='user')
    ax3.scatter(usr_coords[0],usr_coords[1],                            s=10, c='r', marker="o", label='user')
    ax4.scatter(usr_coords[0],usr_coords[1],                            s=10, c='r', marker="o", label='user')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    
    plt.show()
    
    
    
    
    
    
    
    
    print("ECCOLO")
    """
    

    
#us.run_experiment('custom_configs/facebook_best_cf_lgcn.yml',True)


print("END")