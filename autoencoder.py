from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU

class Autoenc:
    def __init__(self,n_components,  epochs=40, batch_size=64, validation_split=0.25):
        self.epochs=epochs 
        self.batch_size=batch_size  
        self.validation_split=validation_split  
        self.n_components=n_components  
    
    def fit_transform(self,data):
        self.input_shape=data.shape
        self.build(data)
        return self.predict(data)
    
    def build_3(self,data):
        input = Input(shape=self.input_shape[1:])
        enc = Dense(self.input_shape[1])(input)
        #enc = LeakyReLU()(enc)
        
        if self.input_shape[1]/2 > self.n_components:
            enc = Dense(int(self.input_shape[1]/2))(enc)
            #enc = LeakyReLU()(enc)     
                
        
        latent_space = Dense(self.n_components, activation="tanh")(enc) 

        if self.input_shape[1]/2 > self.n_components:

            dec = Dense(int(self.input_shape[1]/2))(latent_space)
            #dec = LeakyReLU()(dec)

            
            dec = Dense(units=self.input_shape[1], activation="sigmoid")(dec)     
        
        else:
            
            dec = Dense(self.input_shape[1])(latent_space)
            #dec = LeakyReLU()(dec)
            dec = Dense(units=self.input_shape[1], activation="sigmoid")(dec)    

        autoencoder = Model(input, dec)
        # compile model
        autoencoder.compile(optimizer = "adam",metrics=["mse"],loss="mse")
        # train model        
        autoencoder.fit(data, data,epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)        
        self.encoder = Model(input, latent_space)
    
    def build(self,data):
        input = Input(shape=self.input_shape[1:])
        enc = Dense(self.input_shape[1])(input)
        #enc = LeakyReLU()(enc)
        
        if self.input_shape[1]/2 > self.n_components:
            next_c=int(self.input_shape[1]/2)
            while next_c>self.n_components:
                enc = Dense(next_c)(enc)
                #enc = LeakyReLU()(enc)                
                next_c=int(next_c/2)
                
        
        latent_space = Dense(self.n_components, activation="tanh")(enc) 

        if self.n_components*2 <self.input_shape[1]:
            next_c=self.n_components*2
            dec = Dense(next_c)(latent_space)
            #dec = LeakyReLU()(dec)
            next_c=next_c*2
            while next_c < self.input_shape[1]:
                next_c=next_c*2
                dec = Dense(next_c)(dec)
                #dec = LeakyReLU()(dec)
            
            dec = Dense(units=self.input_shape[1], activation="sigmoid")(dec)     
        
        else:
            
            dec = Dense(self.input_shape[1])(latent_space)
            #dec = LeakyReLU()(dec)
            dec = Dense(units=self.input_shape[1], activation="sigmoid")(dec)    

        autoencoder = Model(input, dec)
        # compile model
        autoencoder.compile(optimizer = "adam",metrics=["mse"],loss="mse")
        # train model        
        autoencoder.fit(data, data,epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)        
        self.encoder = Model(input, latent_space)

    def predict (self,data):
        return self.encoder.predict(data)
    
