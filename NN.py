import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib as plt
import pickle
from tqdm import tqdm
"""
with  open(str('models_raw_data\LightGCN_Custom\yahoo_movies\LightGCN_Custom_data@256') , 'rb') as f:
    GU,GI = pickle.load(f)


GU,GI=GU.cpu().detach().numpy(),GI.cpu().detach().numpy()
"""
class MYAutoencoder(nn.Module):    
    def __init__(self,depth=2):
        self.loss=1000
        self.depth=depth
        super().__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder_users=nn.Sequential()
        self.encoder_items=nn.Sequential()  
        if self.depth<=256:
            self.encoder_users.append(nn.Linear(512,256))
            self.encoder_items.append(nn.Linear(512,256))             
        if self.depth<=128:
            self.encoder_users.append(nn.Linear(256,128))
            self.encoder_items.append(nn.Linear(256,128))
        if self.depth<=64:
            self.encoder_users.append(nn.Linear(128,64))
            self.encoder_items.append(nn.Linear(128,64))
        if self.depth<=32:
            self.encoder_users.append(nn.Linear(64,32))
            self.encoder_items.append(nn.Linear(64,32))
        if self.depth<=16:
            self.encoder_users.append(nn.Linear(32,16))
            self.encoder_items.append(nn.Linear(32,16))
        if self.depth<=8:
            self.encoder_users.append(nn.Linear(16,8))
            self.encoder_items.append(nn.Linear(16,8))
        if self.depth<=4:
            self.encoder_users.append(nn.Linear(8,4))
            self.encoder_items.append(nn.Linear(8,4))
        if self.depth<=2:
            self.encoder_users.append(nn.Linear(4,2))
            self.encoder_items.append(nn.Linear(4,2))
           
      
        
        self.criterion= nn.MSELoss()

        self.optimizer=torch.optim.Adam(self.parameters(),lr=0.0001)#,weight_decay=0.000001)
        self.to(self.device)

    def forward(self,gu,gi):
        gu=self.encoder_users(gu)
        gi=self.encoder_items(gi)
        
        #dot_product = torch.matmul( gu.to(model.device), torch.transpose(gi.to(model.device), 0, 1) )#torch.matmul( gu, torch.transpose(gi,-1,0) )
        #decoded=self.decoder(encoded)
        return gu,gi
    

        

def train(GU,GI,depth=2,num_epochs = 16000):
    model=MYAutoencoder(depth=depth)

    GU=torch.tensor(GU).to(model.device)
    GI=torch.tensor(GI).to(model.device)

    dot_product = torch.matmul( GU.to(model.device), torch.transpose(GI.to(model.device), 0, 1) )#torch.matmul( GU, torch.transpose(GI,-1,0) )

    count=0
    old_loss=1000
    
    with tqdm( total = num_epochs) as t:
        for epoch in range(num_epochs): 
            
            out_gu,out_gi= model(GU,GI)     
            dot_p_e = torch.matmul( out_gu.to(model.device), torch.transpose(out_gi.to(model.device), 0, 1) )#torch.matmul( GU, torch.transpose(GI,-1,0) ) 
            loss = model.criterion(dot_product,dot_p_e)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            t.set_postfix_str(str(f'Epoch:{epoch}, Loss:{loss.item():.4f}'))
            t.update()
            
            #NAIVE EARLY STOP            
            if (old_loss-loss)>0.0001:
                old_loss=loss
                counter=0
            elif counter>1500:
                break
            else:
                counter+=1
            
            
            
    
    return out_gu,out_gi

def train_batches(GU,GI,depth=2,num_epochs = 400):
    model=MYAutoencoder(depth=depth)

    GU=torch.tensor(GU).to(model.device)
    GI=torch.tensor(GI).to(model.device)

    dot_product = torch.matmul( GU.to(model.device), torch.transpose(GI.to(model.device), 0, 1) )#torch.matmul( GU, torch.transpose(GI,-1,0) )
    dataloader_user = DataLoader(GU,
                        batch_size=256,
                        shuffle=True)
    dataloader_items = DataLoader(GU,
                        batch_size=256,
                        shuffle=True)
    
    for epoch in range(num_epochs): 
        for users in dataloader_user:
            for items in dataloader_items:
                
                out_gu,out_gi= model(users,items)
                
                dot_product = torch.matmul( users.to(model.device), torch.transpose(items.to(model.device), 0, 1) )
                dot_p_e = torch.matmul( out_gu.to(model.device), torch.transpose(out_gi.to(model.device), 0, 1) )              
                loss = model.criterion(dot_product,dot_p_e)

                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
            
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        #outputs.append((epoch,USER,recon))
    out_gu,out_gi= model(GU,GI)
    return out_gu,out_gi


def experimental_training(GU,GI,EMBEDDINGS,depth=2,num_epochs = 16000):
    model=MYAutoencoder(depth=depth)
    EMBs=EMBEDDINGS
    GU=torch.tensor(GU).to(model.device)
    GI=torch.tensor(GI).to(model.device)
    dot_product = torch.matmul( GU.to(model.device), torch.transpose(GI.to(model.device), 0, 1) )
    
    EMBs=EMBs[::-1]
    EMBs.append(dot_product)
    for EMBS in EMBEDDINGS:
        
        old_loss=1000
        with tqdm( total = num_epochs) as t:
            for epoch in range(num_epochs): 
                
                out_gu,out_gi= model(GU,GI)     
                dot_p_e = torch.matmul( out_gu.to(model.device), torch.transpose(out_gi.to(model.device), 0, 1) )
                loss = model.criterion(EMBS.to(model.device),dot_p_e)

                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                
                t.set_postfix_str(str(f'Epoch:{epoch}, Loss:{loss.item():.4f}'))
                t.update()
                
                #NAIVE EARLY STOP            
                if (old_loss-loss)>0.0003:
                    old_loss=loss
                    counter=0
                elif counter>1000:
                    break
                else:
                    counter+=1
            
            
            
    
    return out_gu,out_gi
    None


"""
for epoch in range(num_epochs):  
    
    for USER in GU:
        loss = 0
        for ITEM in GI:
            recon = model(USER,ITEM)        
            dot_product = torch.matmul( USER, torch.transpose(ITEM,-1,0) )  
            loss += model.criterion(recon,dot_product)
        
        loss=loss/len_gi
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch,USER,recon))
"""