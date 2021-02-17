
import torch 
import imageio
import numpy as np
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision
from tqdm import tqdm
from torchvision.utils import save_image
import sys, os
sys.path.append(".")
from utils.save_outputs import save_reconstructed_images
from loss import final_loss 
from Model import model
from prepare_data import trainset, trainloader, testset, testloader
from utils.config import args
from utils.device_config import dev
from utils.pred import show_image, visualise_output
import os
import matplotlib.pyplot as plt

device = dev
# if args is None:
#     exit()



epochs = args.epochs
z_dim = args.z_dim
grid_images = []
train_loss = []
valid_loss = []
criterion = nn.BCELoss(reduction='sum')

# Training
def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images





def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss


count = 0
print('[INFO] Training Started')
for epoch in range(epochs):
    count = count+1
    if count == epochs:
        print('[INFO] training about to get finished')
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model.model, trainloader, trainset, device, model.optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model.model, testloader, testset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

torch.save(model.model.state_dict(),\
os.path.join("./", 'model_weight_MNIST.pt'))
print(f'[INFO] Model saved')
i = input('To Plot the predictions? [Enter p]')
images, labels = iter(testloader).next()
if i == 'p':
    print('Original images')
    
    show_image(torchvision.utils.make_grid(images[1:100],10,2))
    plt.show()

    print('VAE reconstruction:')
    visualise_output(images[1:100], model.model)

else:
    pass