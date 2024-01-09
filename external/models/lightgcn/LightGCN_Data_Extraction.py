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
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

#EXPERIMENTAL
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
import umap

class LightGCN(RecMixin, BaseRecommenderModel):
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
        
        self.tsne_sizes = params.meta.tsne_sizes
        self.pca_sizes = params.meta.pca_sizes
        self.kpca_sizes = params.meta.kpca_sizes
        
        self.recs={}
        self.recs['tsne']={'validation':{},'test':{}}
        self.recs['base']={'validation':{},'test':{}}
        self.recs['pca']={'validation':{},'test':{}}
        self.recs['kpca']={'validation':{},'test':{}}
        self.recs['lle']={'validation':{},'test':{}}
        self.recs['isomap']={'validation':{},'test':{}}
        self.recs['umap']={'validation':{},'test':{}}

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
        

        """
        #Save data 
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_data', 'wb')
        pickle.dump([self._data], file)
        file.close()
        
        #Save recs Base
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_base_recs', 'wb')
        pickle.dump(self.recs['base'], file)
        file.close()   
        
        #Save recs PCA
        self.get_recommendations_PCA(self.evaluator.get_needed_recommendations())
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_PCA_recs', 'wb')
        pickle.dump(self.recs['pca'], file)
        file.close()   

        #Save recs Kernel PCA
        self.get_recommendations_KPCA(self.evaluator.get_needed_recommendations())
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_KPCA_recs', 'wb')
        pickle.dump(self.recs['kpca'], file)
        file.close()  
        
        #Save recs TSNE
        self.get_recommendations_TSNE(self.evaluator.get_needed_recommendations())
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_tsne_recs', 'wb')
        pickle.dump(self.recs['tsne'], file)
        file.close()   
        
        #Save recs LLE
        self.get_recommendations_LLE(self.evaluator.get_needed_recommendations())
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_lle_recs', 'wb')
        pickle.dump(self.recs['lle'], file)
        file.close()  
       
        #Save recs ISOMAP
        self.get_recommendations_ISOMAP(self.evaluator.get_needed_recommendations())
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_isomap_recs', 'wb')
        pickle.dump(self.recs['isomap'], file)
        file.close()  
        """
        #Save recs UMAP
        self.get_recommendations_UMAP(self.evaluator.get_needed_recommendations())
        file = open('models_raw_data/'+str(self.__class__.__name__)+'_umap_recs', 'wb')
        pickle.dump(self.recs['umap'], file)
        file.close()  

    def get_recommendations(self, k: int = 100):
          
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
           
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        
        #Update Recommandations Dictionary
        self.recs['base']['validation'][self._batch_size]   =   predictions_top_k_val
        self.recs['base']['test'][self._batch_size]         =   predictions_top_k_test

        return predictions_top_k_val, predictions_top_k_test
    
    def get_recommendations_TSNE(self, k: int = 100):
        
        #Item User concatenztion Tsne Trasformation
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        gu, gi = gu.cpu().detach().numpy(),gi.cpu().detach().numpy()
       
        for n_comp in tqdm(self.tsne_sizes,desc='TSNE iterations'):  #tsne_sizes defined in configs.yml            
            
            if(n_comp>3):
                tsne = TSNE(n_components=n_comp, random_state=42,method='exact')    
            else:
                tsne = TSNE(n_components=n_comp, random_state=42)    
            

                                    
            tnse_predictions_val  = {}
            tnse_predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat = tsne.fit_transform(np.concatenate((gu, gi)))
            u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
            i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
        
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(u_tsne[offset: offset_stop], i_tsne)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                tnse_predictions_val.update(recs_val)
                tnse_predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['tsne']['validation'][n_comp] =   tnse_predictions_val
            self.recs['tsne']['test'][n_comp]       =   tnse_predictions_test
    
    def get_recommendations_PCA(self, k: int = 100):        

        #Item User concatenztion Tsne Trasformation
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        gu, gi = gu.cpu().detach().numpy(),gi.cpu().detach().numpy()
       
        for n_comp in tqdm(self.pca_sizes,desc='PCA iterations'):  #tsne_sizes defined in configs.yml            

            pca = PCA(n_components=n_comp)  
                                                
            pca_predictions_val  = {}
            pca_predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat = pca.fit_transform(np.concatenate((gu, gi)))
            u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
            i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
        
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(u_tsne[offset: offset_stop], i_tsne)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                pca_predictions_val.update(recs_val)
                pca_predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['pca']['validation'][n_comp] =   pca_predictions_val
            self.recs['pca']['test'][n_comp]       =   pca_predictions_test
   
    def get_recommendations_LLE(self, k: int = 100):        
       
        #Item User concatenztion Tsne Trasformation
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        gu, gi = gu.cpu().detach().numpy(),gi.cpu().detach().numpy()
        concatenation = np.concatenate((gu, gi))
        
        self.lln_neighbors=[15,k,len(gi),len(concatenation)-1]
        
        for n_neighbors in tqdm(self.lln_neighbors,desc='LLE iterations'):  #tsne_sizes defined in configs.yml            

            embedding = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2) #N COMPONENTS 2          
                                                
            predictions_val  = {}
            predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat = embedding.fit_transform(concatenation)
            u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
            i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
        
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(u_tsne[offset: offset_stop], i_tsne)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_val.update(recs_val)
                predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['lle']['validation'][n_neighbors] =   predictions_val
            self.recs['lle']['test'][n_neighbors]       =   predictions_test
       
    def get_recommendations_KPCA(self, k: int = 100):        

        #Item User concatenztion Tsne Trasformation
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        gu, gi = gu.cpu().detach().numpy(),gi.cpu().detach().numpy()
       
        for n_comp in tqdm(self.kpca_sizes,desc='KPCA iterations'):  #tsne_sizes defined in configs.yml            

            kpca = KernelPCA(n_components=n_comp)  
                                                
            kpca_predictions_val  = {}
            kpca_predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat = kpca.fit_transform(np.concatenate((gu, gi)))
            u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
            i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
        
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(u_tsne[offset: offset_stop], i_tsne)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                kpca_predictions_val.update(recs_val)
                kpca_predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['kpca']['validation'][n_comp] =   kpca_predictions_val
            self.recs['kpca']['test'][n_comp]       =   kpca_predictions_test
   
    def get_recommendations_ISOMAP(self, k: int = 100):        
       
        #Item User concatenztion Tsne Trasformation
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        gu, gi = gu.cpu().detach().numpy(),gi.cpu().detach().numpy()
        concatenation = np.concatenate((gu, gi))
        
        self.isomap_neighbors=[15,k,len(gi),len(concatenation)-1]
        
        for n_neighbors in tqdm(self.isomap_neighbors,desc='ISOMAP iterations'):  #tsne_sizes defined in configs.yml            

            embedding = Isomap(n_neighbors=n_neighbors, n_components=2) #N COMPONENTS 2          
                                                
            predictions_val  = {}
            predictions_test = {}
        
            #Trasform Concatenated Data
            i_u_concat = embedding.fit_transform(concatenation)
            u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
            i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
        
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(u_tsne[offset: offset_stop], i_tsne)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_val.update(recs_val)
                predictions_test.update(recs_test)

            #Update Recommandations Dictionary
            self.recs['isomap']['validation'][n_neighbors] =   predictions_val
            self.recs['isomap']['test'][n_neighbors]       =   predictions_test
    
    def get_recommendations_UMAP(self, k: int = 100):        
        #UMAP INLY AT 2
        #Item User concatenztion Tsne Trasformation
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        gu, gi = gu.cpu().detach().numpy(),gi.cpu().detach().numpy()
        concatenation = np.concatenate((gu, gi))
        
        self.isomap_neighbors=[15,k,len(gi),len(concatenation)-1]
        
        reducer = umap.UMAP()         
                                                
        predictions_val  = {}
        predictions_test = {}
    
        #Trasform Concatenated Data
        i_u_concat = reducer.fit_transform(concatenation)
        u_tsne =   torch.Tensor(i_u_concat[:self._num_users,:])
        i_tsne =   torch.Tensor( i_u_concat[self._num_users:,:])
    
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(u_tsne[offset: offset_stop], i_tsne)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_val.update(recs_val)
            predictions_test.update(recs_test)

        #Update Recommandations Dictionary
        self.recs['umap']['validation'][2] =   predictions_val
        self.recs['umap']['test'][2]       =   predictions_test

       
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