from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--config', type=str, default='movielens_best_kgtore')
args = parser.parse_args()

#run_experiment(f"config_files/{args.config}.yml")
run_experiment(f"config_files/facebook_lgcn_best_cf.yml")

