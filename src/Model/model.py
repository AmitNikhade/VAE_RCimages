
import torch
import numpy as np
print(torch.__version__)
print(np.__version__)
import torch.nn as nn
from utils.device_config import dev
import torch.optim as optim
from utils.config import args



z_dim = args.z_dim

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        #encoder
        self.conv1 = nn.Conv2d(1 ,8 ,4 ,stride =2 ,padding =1 )
        self.BN1 = nn.BatchNorm2d(8)
        self.af1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8 ,16 ,4 ,stride =2 ,padding = 1)
        self.BN2 = nn.BatchNorm2d(16)
        self.af2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(16 ,32 ,4 ,stride =2 ,padding = 1)
        self.BN3 = nn.BatchNorm2d(32)
        self.af3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(32 ,64 ,4 ,stride =2 ,padding = 0)
        self.BN4 = nn.BatchNorm2d(64)
        self.af4 = nn.LeakyReLU()

        # bottleneck part
        # fully connected layers providing the mean and log variance value 
        self.fc1 = nn.Linear(64,128)
        self.fc_mu = nn.Linear(128, z_dim)
        self.fca1 = nn.LeakyReLU()
        self.fcd1 = nn.Dropout(0.2)
        
        self.fc_log_var= nn.Linear(128, z_dim)
        self.fca2 = nn.LeakyReLU()
        self.fcd2 = nn.Dropout(0.2)
        
        #decoder

        self.fc2 = nn.Linear(z_dim, 64)
        self.da1 = nn.LeakyReLU()
        self.dd1 = nn.Dropout(0.2)

        self.deu1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec1 = nn.ConvTranspose2d(64 ,64 ,4 ,stride =2 ,padding = 0)
        self.deb1 = nn.BatchNorm2d(64)
        self.dea1 = nn.LeakyReLU()
        self.deu2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec2 = nn.ConvTranspose2d(64 ,32 ,4 ,stride =2 ,padding = 1)
        self.deb2 = nn.BatchNorm2d(32)
        self.dea2 = nn.LeakyReLU()
        self.deu3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec3 = nn.ConvTranspose2d(32 ,16 ,4 ,stride =2 ,padding = 1)
        self.deb3 = nn.BatchNorm2d(16)
        self.dea3 = nn.LeakyReLU()
        self.deu4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec4 = nn.ConvTranspose2d(16 ,1 ,4 ,stride =2 ,padding = 1)
        self.dea4 = nn.Sigmoid()


    def sampling(self, mu, log_var):
        std = torch.exp(log_var / 2)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std


    def forward(self, x):
        #creating encoder
        X = self.conv1(x)
        x = self.BN1(X)
        x = self.af1(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.af2(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.af3(x)
        x = self.conv4(x)
        x = self.BN4(x)
        x = self.af4(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        mu = self.fc_mu(x)
        mu = self.fca1(mu)
        mu = self.fcd1(mu)
        log_var = self.fc_log_var(x)
        log_var = self.fca2(log_var)
        log_var = self.fcd2(log_var)
        #creating sampling
        z = self.fc2(self.sampling(mu, log_var))
        z = self.da1(z)
        z = self.dd1(z)
        z = z.view(-1,64,1,1)
        #creating decoder
        d = self.dec1(z)
        d = self.deb1(d)
        d = self.dea1(d)
        d = self.dec2(d)
        d = self.deb2(d)
        d = self.dea2(d)
        d = self.dec3(d)
        d = self.deb3(d)
        d = self.dea3(d)
        d = self.dec4(d)
        recontruction = self.dea4(d)
        return recontruction, mu, log_var

lr = args.learn_rate
device = dev
print('[INFO] using device',device)
model = VAE().to(device)
print('** Variation Autoencoder Summary **\n',model)
optimizer = optim.Adam(model.parameters(), lr=lr)
print('[INFO] learning_rate', optimizer)
