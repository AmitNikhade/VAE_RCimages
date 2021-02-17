
from torchvision.utils import save_image

def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"./output{epoch}.jpg")
    print('reconstruction_saved..')
