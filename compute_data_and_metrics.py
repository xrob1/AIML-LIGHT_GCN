import compute_processing_data as cp
from elliot.run import run_experiment

#Avvia training e salva componenti per l'analisi
#!!!!(ATTENZIONE MOLTO LENTO)!!!!

#cp.load_data('custom_configs/facebook_best_cf_lgcn.yml')

#Prende dizionario con tutte le raccomandazioni fra i vari metodi
#Salvataggio Raccomandazioni validation_set per ogni modalità
cp.save_recs('base',cp.get_base_recs()['validation'])
cp.save_recs_at('tsne',cp.get_tsne_recs()['validation'],2)
cp.save_recs_at('pca' ,cp.get_pca_recs ()['validation'],2)
cp.save_recs_at('kpca',cp.get_kpca_recs()['validation'],2)
cp.save_recs('lle',cp.get_lle_recs()['validation'])
cp.save_recs('isomap',cp.get_isomap_recs()['validation'])
cp.save_recs_at('umap',cp.get_umap_recs()['validation'],2)

#Avvio calcolo metriche, 
run_experiment(f"custom_configs/tsne_metrics.yml")
run_experiment(f"custom_configs/pca_metrics.yml")
run_experiment(f"custom_configs/kpca_metrics.yml")
run_experiment(f"custom_configs/lle_metrics.yml")
run_experiment(f"custom_configs/isomap_metrics.yml")
run_experiment(f"custom_configs/umap_metrics.yml")