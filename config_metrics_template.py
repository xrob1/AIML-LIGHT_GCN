METRICS_TEMPLATE = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data_dz/validation/facebook_base_@64.tsv
    validation_path: ../data_dz/validation/facebook_tsne_@4.tsv
    test_path: ../data_dz/test/facebook_tsne_@4.tsv
  dataset: {dataset}
  top_k: 10
  evaluation:
    cutoffs: [10]
    paired_ttest: True
    simple_metrics: [nDCGRendle2020, HR, Precision, Recall]
  gpu: 0
  models:
    RecommendationFolder:  
        folder: {recs}
"""
