import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])
# training set and train data loader
print('[INFO] downloading and transforming data')
trainset = torchvision.datasets.MNIST(
    root='./', train=True, download=True, transform=transform
)
trainloader = DataLoader(
    trainset, batch_size=64, shuffle=True
)

# validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root='./', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=64, shuffle=False
)