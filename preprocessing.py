from src.preprocessing.preprocessing import run
from src import *


if __name__ == '__main__':

    run(dataset_name=FACEBOOK, core=5, binarize=False)
    run(dataset_name=YAHOO, core=5, threshold=3)
    # run(dataset_name=MOVIELENS, core=5, threshold=3)
