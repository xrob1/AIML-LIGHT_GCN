import compute_processing_data as cp
from elliot.run import run_experiment
from src import *
from src.loader.paths import *
from src.utils.resutls_tab import *

FACTORS=[512,256,128,64,32,16,8,4,2]
DATASET =  MOVIELENS 
ALG = 'LGCN'
REDS=['NN','UMAP','KPCA','AUTOE','TSNE','BASE']

CONFIG_FILE= movielens_exp_conf_file(ALG)#basic_conf_file('BPRMF')#yahoo_exp_conf_file()
configuration = cp.open_config(CONFIG_FILE)#basic_conf_file('BPRMF'))#GET CONFIGURATION TEMPLATE



#EXTRACT RACOMMANDATIONS FROM MODEL WITH REDUCED FACTORS
def get_reduced_factors_model_recs(tipo,FACTORS,configuration,DATASET):
    configuration=set_reducers_configuration(tipo,configuration,[],[],[])
    for f in FACTORS:
        configuration = set_model_factors(tipo,configuration,f)     
        configuration = set_model_weights(tipo,configuration,f,DATASET)  
        cp.save_config( CONFIG_FILE=CONFIG_FILE,configuration=configuration)
        cp.load_data(CONFIG_FILE)

#EXTRACT RACOMMANDATIONS FROM MODEL USING DIFFERENT DIMENSIONALITY REDUCTION METHODS
def get_dimensionatity_reduced_recommandations(tipo,FACTORS,configuration,DATASET):
    configuration = set_model_factors(tipo,configuration,FACTORS[0]) 
    configuration = set_reducers_configuration(tipo,configuration,REDS,FACTORS[1:], ['linear', 'poly','rbf', 'sigmoid','cosine']) 
    #configuration = set_model_weights(tipo,configuration,FACTORS[0],DATASET)  
    cp.save_config( CONFIG_FILE=CONFIG_FILE,configuration=configuration)
    cp.load_data(CONFIG_FILE)

def extract_recs_from_raw_data(type='validation'):
    files = raw_files_names_list(DATASET=DATASET)
    for f in files:
        if f.split("_")[2] == 'KPCA':
            cp.save_recs(f, DATASET,type=type,KPCA=True)
        else:
            cp.save_recs(f, DATASET,type=type)

def clear_results_dir():
    cp.clear_results_dir(DATASET)

def compute_metrics_on_recs(DATASET):
    configuration = cp.open_config(metrics_configuration_file())
    cofiguration= set_runtime_metrics_configuration(configuration,DATASET)
    cp.save_config( CONFIG_FILE=metrics_configuration_file(),configuration=configuration)
    run_experiment(metrics_configuration_file())

def results_csv(name):
    cp.write_results_csv(DATASET=DATASET,out_f_name=name)




#CHANGE CONFIGURATION PARAMETERS TO SUITE EXPERIMENT
#configuration = set_dataset_configuration(configuration, DATASET,'BPRMF')
get_reduced_factors_model_recs(ALG,FACTORS=FACTORS[1:],configuration=configuration,DATASET=DATASET)
#get_dimensionatity_reduced_recommandations(ALG,FACTORS,configuration,DATASET)
#extract_recs_from_raw_data(type='validation')
#clear_results_dir()
#compute_metrics_on_recs(DATASET)
#results_csv('BPRMF_RESULTS_Yah_')



"""
if(INSIDE_METRICS):

    if (CLEAR_RESULTS):
        files = glob.glob(
            'results/' + DATASET + '/performance/*')  # [f for f in os.listdir('results/facebook_book/performance/')]
        for f in files:
            os.remove(f)

    for type in REDS:#['AUTOE','BASE', 'KPCA',  'TSNE', 'UMAP']:
        with open('custom_configs\custom_metrics_runtime.yml', 'r') as file:
            configs = yaml.load(file)
        configs['experiment']['data_config']['train_path'] = '../data/' + DATASET + '/train.tsv'
        configs['experiment']['data_config']['validation_path'] = '../data_dz/' + DATASET + '/base/BASE@256.tsv'
        configs['experiment']['data_config']['test_path'] = '../data_dz/' + DATASET + '/base/BASE@256.tsv'
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
                    line = line.replace('.',',')
                    line=line.split('\t')

                    method = line[0].split('@')[0]
                    n = line[0].split('@')[1]

                    results.append([int(n), method, line[1][:6], line[2][:6], line[3][:6], line[4][:6]])

        results = sorted(results, reverse=True)

        with open('risultati_' + DATASET + '_INSIDE.csv', 'w') as csvfile:

            csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n')

            csvwriter.writerow(['|e|', 'Metodo', 'nDCGRendle2020', 'HR', 'Precision', 'Recall'])

            csvwriter.writerows(results)

        print("stop")
"""   