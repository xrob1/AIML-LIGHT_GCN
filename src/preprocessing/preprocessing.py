from src.preprocessing.filters.basic import IterativeKCore
from src.preprocessing.filters.dataset import Splitter, Binarize, ZeroIndexing
from src.preprocessing.filters.filter import load_dataset, store_dataset
from src.loader.paths import *


def run(dataset_name, core, threshold=None, binarize=True):
    print(f'\n***** {dataset_name} data preprocessing *****\n'.upper())

    dataset_path = dataset_filepath(dataset_name, type='raw')
    dataset = load_dataset(dataset_path)
    print(f'Dataset loaded from {dataset_path}')

    binarized_dataset = dataset

    if binarize:
        if threshold:
            print(f'\n***** {dataset_name} binarization *****\n'.upper())

            binarizer = Binarize(dataset=dataset, threshold=threshold)
            binarized_dataset = binarizer.filter()['dataset']

    # ITERATIVE K-CORE
    print(f'\n***** {dataset_name} iterative k-core *****\n'.upper())
    kcore = IterativeKCore(dataset=binarized_dataset,
                           core=core,
                           kcore_columns=['u', 'i'])
    filtered_dataset = kcore.filter()['dataset']

    # RE-INDEXING
    print(f'*** Reindexing {dataset_name} ***')
    indexer = ZeroIndexing(dataset=filtered_dataset)
    filtered_dataset = indexer.filter()['dataset']

    # SPLITTING
    print(f'\n***** {dataset_name} train-test splitting *****\n'.upper())

    print(f'\nTransactions: {len(filtered_dataset)}')
    print('\nThere will be the splitting...')

    splitter = Splitter(data=filtered_dataset,
                        val_ratio=0.1,
                        test_ratio=0.2)

    splitting_results = splitter.filter()
    print(f'Final training set transactions: {len(splitting_results["train"])}')
    print(f'Final test set transactions: {len(splitting_results["test"])}')
    print(f'Final validation set transactions: {len(splitting_results["val"])}')

    # STORE ON FILE
    data_folder = dataset_directory(dataset_name=dataset_name)

    store_dataset(data=filtered_dataset,
                  folder=data_folder,
                  name=DATASET_NAME,
                  message='filtered dataset')

    store_dataset(data=splitting_results["train"],
                  folder=data_folder,
                  name=TRAIN_NAME,
                  message='training set')

    store_dataset(data=splitting_results["test"],
                  folder=data_folder,
                  name=TEST_NAME,
                  message='test set')

    store_dataset(data=splitting_results["val"],
                  folder=data_folder,
                  name=VAL_NAME,
                  message='val set')
