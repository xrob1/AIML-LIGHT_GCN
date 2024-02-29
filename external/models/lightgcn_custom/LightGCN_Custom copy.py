from tqdm import tqdm
import numpy as np
import torch
import os
import math
import pickle

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .LightGCNModel import LightGCNModel

from torch_sparse import SparseTensor

from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import umap
from autoencoder import Autoenc
import NN as MYAutoencoder

class LightGCN_Custom(RecMixin, BaseRecommenderModel):
    r"""
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3397271.3401063>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        n_layers: Number of stacked propagation layers

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        LightGCN:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          batch_size: 256
          l_w: 0.1
          n_layers: 2
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._sampler = cs.Sampler(self._data.i_train_dict)
        if self._batch_size < 1:
            self._batch_size = self._num_users
        
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_n_layers", "n_layers", "n_layers", 1, int, None)
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        self._model = LightGCNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            n_layers=self._n_layers,
            adj=self.adj,
            random_seed=self._seed
        )
        
        self.reducers_factors =  params.meta.reducers_factors
        self.tsne_reducers_factors =  params.meta.tsne_reducers_factors
        self.reducers =  params.meta.reducers_types
        self.KPCA_Kernels =  params.meta.KPCA_Kernels
        self.dataset_name =  params.meta.dataset_name
        
        
        self.recs={}

    @property
    def name(self):
        return "LightGCN" \
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
        file = open('models_raw_data/LightGCN_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_BASE_recs@'+str(self._factors), 'wb')
        pickle.dump(self.recs['base'], file)
        file.close()  

        if 'NN' in self.reducers :
            self.get_recommendations_NN(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/LightGCN_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_NN_recs', 'wb')
            pickle.dump(self.recs['NN'], file)
            file.close()
        
        if 'AUTOE' in self.reducers :        
            #Save recs PCA
            self.get_recommendations_AUTOE(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/LightGCN_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_AUTOE_recs', 'wb')
            pickle.dump(self.recs['AUTOE'], file)
            file.close()   
        
        if 'KPCA' in self.reducers :  
            #Save recs Kernel PCA
            self.get_recommendations_KPCA(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/LightGCN_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_KPCA_recs', 'wb')
            pickle.dump(self.recs['KPCA'], file)
            file.close()  
        
        if 'TSNE' in self.reducers :  
        #Save recs TSNE
            self.get_recommendations_TSNE(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/LightGCN_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_TSNE_recs', 'wb')
            pickle.dump(self.recs['TSNE'], file)
            file.close()   
        
        if 'UMAP' in self.reducers :  
            #Save recs/data UMAP
            self.get_recommendations_UMAP(self.evaluator.get_needed_recommendations())
            file = open('models_raw_data/LightGCN_Custom/'+self.dataset_name+'/'+str(self.__class__.__name__)+'_UMAP_recs', 'wb')
            pickle.dump(self.recs['UMAP'], file)
            file.close() 

        
    def extract_recs_from_embeddings(self,users,items,k: int = 100):
        preds_test={}
        preds_val={}
        
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(users[offset: offset_stop], items)           
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            preds_test.update(recs_val)
            preds_val.update(recs_test)
        
        return preds_val,preds_test
    
    def get_recommendations(self, k: int = 100):
        self.recs['base']={'validation':{},'test':{}}

        gu, gi = self._model.propagate_embeddings(evaluate=True)
        
        self.GU,self.GI= gu.cpu().detach().numpy(),gi.cpu().detach().numpy()
        
        predictions_top_k_val,predictions_top_k_test = self.extract_recs_from_embeddings(gu,gi,k)
      
        #Update Recommandations Dictionary
        self.recs['base']['validation'][self._factors]   =   predictions_top_k_val
        self.recs['base']['test'][self._factors]         =   predictions_top_k_test

        return predictions_top_k_val, predictions_top_k_test
        
    def get_recommendations_TSNE(self, k: int = 100):
        self.recs['TSNE']={'validation':{},'test':{}}
        
        gu, gi = self.GU,self.GI              
            
        tsne = TSNE(n_components=2, random_state=42) 
        i_u_concat = tsne.fit_transform(np.concatenate((gu, gi)))
        gu_r =   torch.Tensor(i_u_concat[:self._num_users,:])
        gi_r =   torch.Tensor( i_u_concat[self._num_users:,:])
    
        predictions_top_k_val, predictions_top_k_test = self.extract_recs_from_embeddings(gu_r,gi_r,k)

        #Update Recommandations Dictionary
        self.recs['TSNE']['validation'][2] =   predictions_top_k_val
        self.recs['TSNE']['test'][2]       =   predictions_top_k_test
    
    def get_recommendations_AUTOE(self, k: int = 100):        
        self.recs['AUTOE']={'validation':{},'test':{}}
        #Item User concatenztion Tsne Trasformation
        gu, gi =self.GU,self.GI# self._model.propagate_embeddings(evaluate=True)

       
        for n_components in tqdm(self.reducers_factors,desc='AUTOE iterations'):  #tsne_sizes defined in configs.yml           

            autoenc = Autoenc(n_components=n_components)          
            #Trasform Concatenated Data
            i_u_concat = autoenc.fit_transform(np.concatenate((gu, gi)))
            gu_r =   torch.Tensor(i_u_concat[:self._num_users,:])
            gi_r =   torch.Tensor( i_u_concat[self._num_users:,:])
        
            predictions_top_k_val,predictions_top_k_test = self.extract_recs_from_embeddings(gu_r,gi_r,k)

            #Update Recommandations Dictionary
            self.recs['AUTOE']['validation'][n_components] =   predictions_top_k_val
            self.recs['AUTOE']['test'][n_components]       =   predictions_top_k_test
   
    def get_recommendations_KPCA(self, k: int = 100):     


        gu, gi = self.GU,self.GI

        self.recs['KPCA']={'validation':{},'test':{}}
        for kernel in self.KPCA_Kernels:
            self.recs['KPCA']['validation'][kernel]={}
            self.recs['KPCA']['test'][kernel]={}
            for n_components in tqdm(self.reducers_factors,desc='KPCA-'+str(kernel)+' iterations'):  #tsne_sizes defined in configs.yml            

                kpca = KernelPCA(n_components=n_components,kernel=kernel)                                                     
            
                #Trasform Concatenated Data
                i_u_concat = kpca.fit_transform(np.concatenate((gu, gi)))
                
                gu_r =   torch.Tensor(i_u_concat[:self._num_users,:])
                gi_r =   torch.Tensor( i_u_concat[self._num_users:,:])
            
                predictions_top_k_val, predictions_top_k_test = self.extract_recs_from_embeddings(gu_r,gi_r,k)

                #Update Recommandations Dictionary
                self.recs['KPCA']['validation'][kernel][n_components] =   predictions_top_k_val
                self.recs['KPCA']['test'][kernel][n_components]       =   predictions_top_k_test
   
    def get_recommendations_UMAP(self, k: int = 100):        
        #UMAP ONLY AT 2 DIM
        #Item User concatenztion Tsne Trasformation
        gu, gi = self.GU,self.GI#self._model.propagate_embeddings(evaluate=True)
        
        self.recs['UMAP']={'validation':{},'test':{}}
        self.umap_data = {}
        
        for n_components in tqdm(self.reducers_factors,desc='UMAP iteration'):    

            reducer = umap.UMAP(n_components=n_components)         
        
            #Trasform Concatenated Data
            i_u_concat =torch.Tensor( reducer.fit_transform(np.concatenate((gu, gi))))
            gu_r =    i_u_concat[:self._num_users,:] 
            gi_r =    i_u_concat[self._num_users:,:] 
        
            predictions_top_k_val, predictions_top_k_test = self.extract_recs_from_embeddings(gu_r,gi_r,k)

            #Update Recommandations Dictionary
            self.recs['UMAP']['validation'][n_components] =   predictions_top_k_val
            self.recs['UMAP']['test'][n_components]       =   predictions_top_k_test            

    def get_recommendations_NN(self, k: int = 100):
        self.recs['NN']={'validation':{},'test':{}}

        GU,GI=self.GU,self.GI
        
        for n_components in tqdm(self.reducers_factors ,desc='NN iterations'): 

            gu_r,gi_r = MYAutoencoder.train(GU,GI,depth=n_components)
            predictions_top_k_val, predictions_top_k_test = self.extract_recs_from_embeddings(gu_r,gi_r,k)           
            #Update Recommandations Dictionary
            self.recs['NN']['validation'][n_components]   =   predictions_top_k_val
            self.recs['NN']['test'][n_components]         =   predictions_top_k_test

          
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
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
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