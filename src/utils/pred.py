import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from utils.device_config import dev

device = dev
def to_img(x):
    x = x.clamp(0, 1)
    return x
def visualise_output(images, model):

    with torch.no_grad():
    
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:100], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()


def show_image(img):
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()
