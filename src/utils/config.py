import argparse

def parse_args():
    desc = "PyTorch implementation of 'Variational AutoEncoder (VAE)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')

    parser.add_argument('--z_dim', type=int, default=20, help='Dimension of latent vector')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    return parser.parse_args()

args = parse_args()