import compute_processing_data as cp
from elliot.run import run_experiment

DATASET = "facebook/"

#Avvia training e salva componenti per l'analisi
#!!!!(ATTENZIONE MOLTO LENTO)!!!!

#cp.load_data()

#Prende dizionario con tutte le raccomandazioni fra i vari metodi
recs = cp.get_recs_dict()

#Salvataggio Raccomandazioni validation_set per ogni modalità
for MODE in ["validation","test"]:
    for METHOD in recs:	
        for SIZE in recs[METHOD][MODE]:		
            path = 'data_dz/facebook/'+METHOD+'/'+MODE+'/'
            name = MODE+'_'+METHOD+'_@'+str(SIZE)+'.tsv'
            
            file = open(path+name, 'w')		
            for USER in recs[METHOD][MODE][SIZE]:
                for rec in recs[METHOD][MODE][SIZE][USER]:
                    file.write(str(str(USER)+"	"+str(rec[0])+"	"+str(rec[1])+'\n'))
            file.close()   

#Avvio calcolo metriche, modificare file  config_files\runtime_metrics_conf.yml
run_experiment(f"config_files/runtime_metrics_conf.yml")
