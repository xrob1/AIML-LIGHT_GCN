import compute_processing_data as cp
from elliot.run import run_experiment



#Avvia training e salva componenti per l'analisi
#!!!!(ATTENZIONE MOLTO LENTO)!!!!
#cp.load_data()

#Prende dizionario con tutte le raccomandazioni fra i vari metodi
#Salvataggio Raccomandazioni validation_set per ogni modalità
cp.save_recs('base',cp.get_base_recs()['validation'])
cp.save_recs('tsne',cp.get_tsne_recs()['validation'])
cp.save_recs('pca' ,cp.get_pca_recs() ['validation'])
cp.save_recs('kpca',cp.get_kpca_recs()['validation'])


#Avvio calcolo metriche, modificare file  config_files\runtime_metrics_conf.yml
run_experiment(f"custom_configs/tsne_metrics.yml")
run_experiment(f"custom_configs/pca_metrics.yml")
run_experiment(f"custom_configs/kpca_metrics.yml")