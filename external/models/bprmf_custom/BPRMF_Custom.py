"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import torch
import os
from tqdm import tqdm
import math

from elliot.dataset.samplers import custom_sampler as cs
from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from .BPRMFModel import BPRMFModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger

from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import umap
from autoencoder import Autoenc
import NN as MYAutoencoder
import pickle
import numpy as np
class BPRMF_Custom(RecMixin, BaseRecommenderModel):
    r"""
    Batch Bayesian Personalized Ranking with Matrix Factorization

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        l_w: Regularization coefficient for latent factors

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        BPRMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.1
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w]: regularization,
                                      lr: learning rate}
        """

        self._params_list = [
            ("_factors", "factors", "factors", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_l_w", "l_w", "l_w", 0.1, float, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = cs.Sampler(self._data.i_train_dict, self._seed)

        self._model = BPRMFModel(self._num_users,
                                 self._num_items,
                                 self._learning_rate,
                                 self._factors,
                                 self._l_w,
                                 self._seed)
        
        self.reducers_factors =  params.meta.reducers_factors
        self.tsne_reducers_factors =  params.meta.tsne_reducers_factors
        self.reducers =  params.meta.reducers_types
        self.KPCA_Kernels =  params.meta.KPCA_Kernels
        self.dataset_name =  params.meta.dataset_name
        
        
        self.recs={}

    @property
    def name(self):
        return "BPRMF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))
            
        #Save recs Base
        file = open('models_raw_data/BPRMF_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_BASE_recs@'+str(self._factors), 'wb')
        pickle.dump(self.recs['base'], file)
        file.close()  

        if 'NN' in self.reducers :
            self.get_recommendations_NN(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/BPRMF_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_NN_recs', 'wb')
            pickle.dump(self.recs['NN'], file)
            file.close()
        
        
        if 'AUTOE' in self.reducers :        
            #Save recs PCA
            self.get_recommendations_AUTOE(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/BPRMF_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_AUTOE_recs', 'wb')
            pickle.dump(self.recs['AUTOE'], file)
            file.close()   
        
        if 'KPCA' in self.reducers :  
            #Save recs Kernel PCA
            self.get_recommendations_KPCA(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/BPRMF_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_KPCA_recs', 'wb')
            pickle.dump(self.recs['KPCA'], file)
            file.close()  
        
        if 'TSNE' in self.reducers :  
        #Save recs TSNE
            self.get_recommendations_TSNE(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/BPRMF_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_TSNE_recs', 'wb')
            pickle.dump(self.recs['TSNE'], file)
            file.close()   
        
        if 'UMAP' in self.reducers :  
            #Save recs/data UMAP
            self.get_recommendations_UMAP(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/BPRMF_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_UMAP_recs', 'wb')
            pickle.dump(self.recs['UMAP'], file)
            file.close() 


    def get_recommendations(self, k: int = 100):
        self.recs['base']={'validation':{},'test':{}}
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        

        
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):            
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        
        #Update Recommandations Dictionary
        self.Gu=self._model.Gu.weight
        self.Gi=self._model.Gi.weight
        self.recs['base']['validation'][self._factors]   =   predictions_top_k_val
        self.recs['base']['test'][self._factors]         =   predictions_top_k_test
        return predictions_top_k_val, predictions_top_k_test

    def get_recommendations_TSNE(self, k: int = 100):
        self.recs['TSNE']={'validation':{},'test':{}}
        #Item User concatenztion Tsne Trasformation
        gu, gi =self.Gu.cpu().detach().numpy(),self.Gi.cpu().detach().numpy()# self._model.propagate_embeddings(evaluate=True)
        #GU, GI = GU.cpu().detach().numpy(),GI.cpu().detach().numpy()
       
        for n_components in tqdm(self.tsne_reducers_factors ,desc='TSNE iterations'):  #tsne_sizes defined in configs.yml            
            
            if(n_components>3):
                tsne = TSNE(n_components=n_components, random_state=42,method='exact')    
            else:
                tsne = TSNE(n_components=n_components, random_state=42)    
            

                                    
            tnse_predictions_val  = {}
            tnse_predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat = tsne.fit_transform(np.concatenate((gu, gi)))
            u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
            i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])

            self._model.Gu.weight = torch.nn.Parameter(u_tsne)
            self._model.Gi.weight = torch.nn.Parameter(i_tsne)
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):            
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                tnse_predictions_val.update(recs_val)
                tnse_predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['TSNE']['validation'][n_components] =   tnse_predictions_val
            self.recs['TSNE']['test'][n_components]       =   tnse_predictions_test
    
    def get_recommendations_AUTOE(self, k: int = 100):        
        self.recs['AUTOE']={'validation':{},'test':{}}
        #Item User concatenztion Tsne Trasformation
        gu, gi =self.Gu.cpu().detach().numpy(),self.Gi.cpu().detach().numpy()# self._model.propagate_embeddings(evaluate=True)
        #GU, GI = GU.cpu().detach().numpy(),GI.cpu().detach().numpy()
       
        for n_components in tqdm(self.reducers_factors,desc='AUTOE iterations'):  #tsne_sizes defined in configs.yml            

            autoenc = Autoenc(n_components=n_components)  
                                                
            autoe_predictions_val  = {}
            autoe_predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat = autoenc.fit_transform(np.concatenate((gu, gi)))
            u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
            i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
            
            self._model.Gu.weight = torch.nn.Parameter(u_tsne)
            self._model.Gi.weight = torch.nn.Parameter(i_tsne)
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):            
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                autoe_predictions_val.update(recs_val)
                autoe_predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['AUTOE']['validation'][n_components] =   autoe_predictions_val
            self.recs['AUTOE']['test'][n_components]       =   autoe_predictions_test
   
    def get_recommendations_KPCA(self, k: int = 100):        

        #Item User concatenztion Tsne Trasformation
        gu, gi =self.Gu.cpu().detach().numpy(),self.Gi.cpu().detach().numpy()#self._model.propagate_embeddings(evaluate=True)
        #GU, GI = GU.cpu().detach().numpy(),GI.cpu().detach().numpy()
        self.recs['KPCA']={'validation':{},'test':{}}
        for kernel in self.KPCA_Kernels:
            self.recs['KPCA']['validation'][kernel]={}
            self.recs['KPCA']['test'][kernel]={}
            for n_components in tqdm(self.reducers_factors,desc='KPCA-'+str(kernel)+' iterations'):  #tsne_sizes defined in configs.yml            

                kpca = KernelPCA(n_components=n_components,kernel=kernel)  
                                                    
                kpca_predictions_val  = {}
                kpca_predictions_test = {}
            
                #Trasform Concatenated Data
                i_u_concat = kpca.fit_transform(np.concatenate((gu, gi)))
                u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
                i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
                self._model.Gu.weight = torch.nn.Parameter(u_tsne)
                self._model.Gi.weight = torch.nn.Parameter(i_tsne)
                for index, offset in enumerate(range(0, self._num_users, self._batch_size)):            
                    offset_stop = min(offset + self._batch_size, self._num_users)
                    predictions = self._model.predict(offset, offset_stop)
                    recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                    kpca_predictions_val.update(recs_val)
                    kpca_predictions_test.update(recs_test)

                #Update Recommandations Dictionary
                self.recs['KPCA']['validation'][kernel][n_components] =   kpca_predictions_val
                self.recs['KPCA']['test'][kernel][n_components]       =   kpca_predictions_test
   
    def get_recommendations_UMAP(self, k: int = 100):        
        #UMAP ONLY AT 2 DIM
        #Item User concatenztion Tsne Trasformation
        gu, gi = self.Gu.cpu().detach().numpy(),self.Gi.cpu().detach().numpy()#self._model.propagate_embeddings(evaluate=True)
        #GU, GI = GU.cpu().detach().numpy(),GI.cpu().detach().numpy()
        
        self.recs['UMAP']={'validation':{},'test':{}}
        self.umap_data = {}
        
        for n_components in tqdm(self.reducers_factors,desc='UMAP iteration'):    

            reducer = umap.UMAP(n_components=n_components)         
                                                    
            predictions_val  = {}
            predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat =torch.Tensor( reducer.fit_transform(np.concatenate((gu, gi))))
            u_tsne =    i_u_concat[:self._num_users,:] 
            i_tsne =    i_u_concat[self._num_users:,:] 
            self._model.Gu.weight = torch.nn.Parameter(u_tsne)
            self._model.Gi.weight = torch.nn.Parameter(i_tsne)

            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):            
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_val.update(recs_val)
                predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['UMAP']['validation'][n_components] =   predictions_val
            self.recs['UMAP']['test'][n_components]       =   predictions_test            
            #self.umap_data[n_components]  =   [ i_u_concat[:self._num_users,:], i_u_concat[self._num_users:,:] ]

    def get_recommendations_NN(self, k: int = 100):
        self.recs['NN']={'validation':{},'test':{}}
        embs = []
        GU,GI=self.Gu,self.Gi
        ##GU, GI = GU.cpu().detach().numpy(),GI.cpu().detach().numpy()
        
        for n_components in tqdm(self.reducers_factors ,desc='NN iterations'):  #tsne_sizes defined in configs.yml   

            out_gu,out_gi = MYAutoencoder.train(GU,GI,depth=n_components)
            self._model.Gu.weight = torch.nn.Parameter(out_gu)
            self._model.Gi.weight = torch.nn.Parameter(out_gi)
            
            preds_test={}
            preds_val={}

            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):            
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                preds_val.update(recs_val)
                preds_test.update(recs_test)
            
            #Update Recommandations Dictionary
            self.recs['NN']['validation'][n_components]   =   preds_val
            self.recs['NN']['test'][n_components]         =   preds_test

        return preds_test, preds_val
   
    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
