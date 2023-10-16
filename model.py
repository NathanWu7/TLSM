import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import reparameterize

# Models for tactile
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)



class Encoder(nn.Module):
    def __init__(self, content_latent_size = 32):
        super(Encoder, self).__init__()
        self.content_latent_size = content_latent_size

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,stride=2,padding=1), ##64x64
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=2,padding=1), ##32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=2,padding=1), ##16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2,padding=1), ##8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(8*8*64, 1024)
        )

        self.fc_mean = nn.Linear(1024, content_latent_size) # 8*8*64
        self.fc_logvar = nn.Linear(1024, content_latent_size)  # 8*8*64
        

    def forward(self, x):
        x = self.main(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_size = 32):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, 1024),  # 8*8*64
            nn.Linear(1024, 8*8*64),  # 8*8*64
            Reshape(64,8,8), # 64，8，8
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3,stride=2,output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3,stride=2,output_padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.main(x)
        return x
        
class FeatureMapping(nn.Module):
    def __init__(self, latent_size = 32):
        super(FeatureMapping, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LeakyReLU(),     
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.LeakyReLU(),   
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LeakyReLU(),  
            nn.Dropout(0.1),  

            nn.Linear(512, 512),
            nn.LeakyReLU(),  
            nn.Dropout(0.1),
            
            nn.Linear(512, latent_size),   
        )
    def forward(self, x):
        x = self.main(x)
        return x    
        
class Discriminator(nn.Module):
    def __init__(self, latent_size = 32):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LeakyReLU(),     
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.LeakyReLU(),   
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LeakyReLU(),  
            nn.Dropout(0.1),  

            nn.Linear(512, 512),
            nn.LeakyReLU(),  
            nn.Dropout(0.1),
            
            nn.Linear(512, 1),   
        )

    def forward(self, x):
        x = self.main(x)
        x = torch.sigmoid(x)
        return x
        
class VAE(nn.Module):
    def __init__(self, content_latent_size = 32):
        super(VAE, self).__init__()
        self.encoder = Encoder(content_latent_size)
        self.decoder = Decoder(content_latent_size) 
       

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        recon_x = self.decoder(contentcode)

        return mu, logsigma, recon_x

