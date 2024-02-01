import compute_processing_data as cp
import pickle
import os
from elliot.run import run_experiment
import glob
import csv
from src import *
from src.loader.paths import CONFIG_DIR, basic_conf_file, dataset_filepath
import ruamel.yaml
import ruamel.yaml.util

from src.utils.resutls_tab import *

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=4, sequence=6, offset=1)


def save_recs(f, DATASET):
    with open('models_raw_data/LightGCN_Custom/' + DATASET + '/' + f, 'rb') as file:
        recs = pickle.load(file)

    for SIZE in recs['validation'].keys():
        with open('data_dz/' + DATASET + '/' + f.split("_")[2] + '/' + f.split("_")[2] + '@' + str(SIZE) + '.tsv',
                  'w') as file:
            for USER in recs['validation'][SIZE]:
                for rec in recs['validation'][SIZE][USER]:
                    file.write(str(str(USER) + "	" + str(rec[0]) + "	" + str(rec[1]) + '\n'))


def save_recs_kpca(f, DATASET):
    with open('models_raw_data/LightGCN_Custom/' + DATASET + '/' + f, 'rb') as file:
        recs = pickle.load(file)

    for KERNEL in recs['validation'].keys():
        for SIZE in recs['validation'][KERNEL].keys():
            with open('data_dz/' + DATASET + '/' + f.split("_")[2] + '/' + f.split("_")[2] + '_' + KERNEL + '@' + str(
                    SIZE) + '.tsv', 'w') as file:
                for USER in recs['validation'][KERNEL][SIZE]:
                    for rec in recs['validation'][KERNEL][SIZE][USER]:
                        file.write(str(str(USER) + "	" + str(rec[0]) + "	" + str(rec[1]) + '\n'))


DATASET =  YAHOO # ['yahoo_movies','facebook_book']

BASE = False
REDUCERS = False

DATA_EXTRACTION = False

METRICS = True
CLEAR_RESULTS = True
CSV = True

with open(basic_conf_file(), 'r') as file:
    configuration = yaml.load(file)

# DATASET CONFIGURATION
configuration = set_dataset_configuration(configuration, DATASET)

# RESULTS FOR BASE AT DIFFERENT FACTORS
if (BASE):
    for factors in [64, 32, 16, 8, 4, 2]:
        # REDUCERS OPTIONS
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['reducers_types'] = []
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['reducers_factors'] = []
        configuration['experiment']['models']['external.LightGCN_Custom']['meta']['KPCA_Kernels'] = ['linear', 'poly',
                                                                                                     'rbf', 'sigmoid',
                                                                                                     'cosine']
        configuration['experiment']['models']['external.LightGCN_Custom']['factors'] = factors
        with open('config_files/custom.yml', 'w') as file:
            yaml.dump(configuration, file)

        cp.load_data('config_files/custom.yml')

if (REDUCERS):
    configuration['experiment']['models']['external.LightGCN_Custom']['factors'] = 64
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['reducers_types'] = ['UMAP', 'PCA',
                                                                                                   'KPCA', 'TSNE']
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['reducers_factors'] = [32, 16, 8, 4, 2]
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['tsne_reducers_factors'] = [2]
    configuration['experiment']['models']['external.LightGCN_Custom']['meta']['KPCA_Kernels'] = ['linear', 'poly',
                                                                                                 'rbf', 'sigmoid',
                                                                                                 'cosine']
    with open('config_files/custom.yml', 'w') as file:
        yaml.dump(configuration, file)
    cp.load_data('config_files/custom.yml')

if (DATA_EXTRACTION):

    files = [f for f in os.listdir('models_raw_data/LightGCN_Custom/' + DATASET + '/')]
    for f in files:
        if (f.split("_")[2] in ['base', 'PCA', 'tsne']) or (f.split("_")[2] == 'umap' and f.split("_")[3] == 'recs'):
            save_recs(f, DATASET)
        if f.split("_")[2] == 'KPCA':
            save_recs_kpca(f, DATASET)

if (METRICS):

    if (CLEAR_RESULTS):
        files = glob.glob(
            'results/' + DATASET + '/performance/*')  # [f for f in os.listdir('results/facebook_book/performance/')]
        for f in files:
            os.remove(f)

    for type in ['base', 'kpca', 'pca', 'tsne', 'umap']:
        with open('custom_configs\custom_metrics_runtime.yml', 'r') as file:
            configs = yaml.load(file)
        configs['experiment']['data_config']['train_path'] = '../data/' + DATASET + '/train.tsv'
        configs['experiment']['data_config']['validation_path'] = '../data/' + DATASET + '/val.tsv'
        configs['experiment']['data_config']['test_path'] = '../data/' + DATASET + '/test.tsv'
        configs['experiment']['models']['RecommendationFolder']['folder'] = os.path.abspath(
            'data_dz/' + DATASET + '/' + type)
        configs['experiment']['dataset'] = DATASET

        with open('custom_configs\custom_metrics_runtime.yml', 'w') as file:
            yaml.dump(configs, file)

        run_experiment('custom_configs\custom_metrics_runtime.yml')

        print("STOP")

if (CSV):

    results = []

    for f in [f for f in os.listdir('results/' + DATASET + '/performance')]:
        if f.split('_')[0] != 'rec': continue
        with open('results/' + DATASET + '/performance/' + f, 'r') as file:

            fields = file.readline()  # SALTA INTESTAZIONE:
            for line in file:
                line = line.split('\t')

                method = line[0].split('@')[0]
                n = line[0].split('@')[1]

                results.append([int(n), method, line[1][:6], line[2][:6], line[3][:6], line[4][:6]])

    results = sorted(results, reverse=True)

    with open('risultati_' + DATASET + '.csv', 'w') as csvfile:

        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n')

        csvwriter.writerow(['|e|', 'Metodo', 'nDCGRendle2020', 'HR', 'Precision', 'Recall'])

        csvwriter.writerows(results)

    print("stop")

print("STOP")
