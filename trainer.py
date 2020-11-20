from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import glob
from typing import Optional, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

PRINT_ITER = 10
TEST_RATIO = .05
IMG_FOLDER = 'CelebA'


def load_split_train_test(batch_size, cpu) -> Tuple[DataLoader, DataLoader]:
    train_transforms = transforms.Compose([transforms.Resize(128),
                                           transforms.ToTensor(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    train_data = datasets.ImageFolder(IMG_FOLDER, transform=train_transforms)
    test_data = datasets.ImageFolder(IMG_FOLDER, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(TEST_RATIO * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               sampler=train_sampler, batch_size=batch_size, num_workers=not cpu,
                                               pin_memory=not cpu)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              sampler=test_sampler, batch_size=batch_size, num_workers=not cpu,
                                              pin_memory=not cpu)
    return train_loader, test_loader


class VAE(nn.Module):
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3*16384))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Trainer:
    def __init__(self, device, epochs, batch_size, cpu: bool):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.cpu = cpu
        self.train_loader, self.test_loader = load_split_train_test(batch_size, cpu)
        self.losses: List[float] = list()
        self.model = VAE().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def plot_train_losses(self):
        """
        :return:
        """
        plt.plot(self.train_losses)
        plt.title('Results')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def plot_train_and_test_losses(self):
        """
        :return:
        """
        plt.plot(self.train_losses)
        plt.plot(self.test_losses)
        plt.legend(['train', 'test'])
        plt.title('Results')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    @staticmethod
    def loss_function_l1(recon_x, x):
        l1_loss = nn.L1Loss()
        return l1_loss(recon_x, x)

    @staticmethod
    def loss_function_l2(recon_x, x):
        l2_loss = nn.MSELoss()
        return l2_loss(recon_x, x)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data[0].to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % PRINT_ITER == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.data.train.dataset),
                           100. * batch_idx / len(self.data.train),
                           loss.item() / len(data)))

        train_loss /= len(self.train_loader.dataset)
        self.train_losses.append(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))

    def test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data[0].to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(self.batch_size, 1, 64, 64)[:n]])
                    save_image(comparison.cpu(),
                               'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print('====> Test set loss: {:.4f}'.format(test_loss))


def run_trainer(trainer):
    for epoch in range(0, trainer.epochs):
        trainer.train(epoch)
        trainer.test_epoch(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(trainer.device)
            sample = trainer.model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

    trainer.plot_train_losses()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='force CPU usage')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    args.cpu = args.cpu or not torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu else "cuda")

    trainer = Trainer(device, epochs=10, batch_size=100, cpu=args.cpu)
    run_trainer(trainer)


if __name__ == "__main__":
    main()
