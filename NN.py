import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class MYAutoencoder(nn.Module):    
    def __init__(self,depth=2,in_shape=512):
        self.loss=1000
        self.depth=depth
        self.in_shape=in_shape
        super().__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder_users=nn.Sequential()
        self.encoder_items=nn.Sequential()  
        if self.depth<=256 and self.in_shape>256:
            self.encoder_users.append(nn.Linear(512,256))
            self.encoder_items.append(nn.Linear(512,256))             
        if self.depth<=128  and self.in_shape>128:
            self.encoder_users.append(nn.Linear(256,128))
            self.encoder_items.append(nn.Linear(256,128))
        if self.depth<=64 and self.in_shape>64:
            self.encoder_users.append(nn.Linear(128,64))
            self.encoder_items.append(nn.Linear(128,64))
        if self.depth<=32 and self.in_shape>32:
            self.encoder_users.append(nn.Linear(64,32))
            self.encoder_items.append(nn.Linear(64,32))
        if self.depth<=16 and self.in_shape>16:
            self.encoder_users.append(nn.Linear(32,16))
            self.encoder_items.append(nn.Linear(32,16))
        if self.depth<=8  and self.in_shape>8:
            self.encoder_users.append(nn.Linear(16,8))
            self.encoder_items.append(nn.Linear(16,8))
        if self.depth<=4  and self.in_shape>4:
            self.encoder_users.append(nn.Linear(8,4))
            self.encoder_items.append(nn.Linear(8,4))
        if self.depth<=2  and self.in_shape>2:
            self.encoder_users.append(nn.Linear(4,2))
            self.encoder_items.append(nn.Linear(4,2))
           
      
        
        self.criterion= nn.MSELoss()

        self.optimizer=torch.optim.Adam(self.parameters(),lr=0.0001)#,weight_decay=0.000001)
        self.to(self.device)

    def forward(self,gu,gi):
        gu=self.encoder_users(gu)
        gi=self.encoder_items(gi)

        return gu,gi
    

        

def train(GU,GI,depth=2,num_epochs = 124000):
    model=MYAutoencoder(depth=depth,in_shape=GU.shape[1])

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
            elif counter>2000:
                break
            else:
                counter+=1
    
    return out_gu,out_gi



