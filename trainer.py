from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import glob
from typing import Optional, List
import matplotlib
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

PRINT_ITER = 10


class CelebLoader:

    @staticmethod
    def to_tensor(img: str):
        pass

    def __init__(self, cpu):
        self.cpu = cpu
        dataset = datasets.ImageFolder('CelebA')
        celebs = glob.iglob("CelebA/*.jpg")
        self.celebs = np.fromiter(celebs, dtype='U18')
        self.train: Optional[List[str]] = None
        self.test: Optional[List[str]] = None

    def partition_train_test(self, batch_size, dataset):
        """
        :return: after this is invoked, train and test should be shuffled and partitioned
        """
        # np.random.shuffle(self.celebs)
        # train_start_idx = int(0.05 * len(self.celebs))
        # self.train = self.celebs[train_start_idx:]
        # self.test = self.celebs[:train_start_idx]

        transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
        dataset = datasets.ImageFolder(dataset, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=not self.cpu,
                                                  pin_memory=not self.cpu)
        images, labels = next(iter(data_loader))

    def train_iter(self):
        return (self.to_tensor(img) for img in self.train)

    def test_iter(self):
        return (self.to_tensor(img) for img in self.test)


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
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Trainer:
    def __init__(self, device, epochs, batch_size, cpu: bool):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.cpu = cpu
        self.data = CelebLoader()
        self.losses: List[float] = list()
        self.model = VAE().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def plot_losses(self):
        """
        do something with self.losses
        :return:
        """
        plt.plot(self.losses)
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

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.data.train):
            data = data.to(self.device)
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

        train_loss /= len(self.data.train.dataset)
        self.train_losses.append(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))

    def test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.data.test):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                               'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs for training')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='force CPU usage')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed')

    args = parser.parse_args()
    args.cpu = args.cpu or not torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu else "cuda")

    trainer = Trainer(device, args.epochs, args.batch_size, args.cpu)

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        trainer.test_epoch(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = trainer.model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

    trainer.plot_losses(trainer.losses)


if __name__ == "__main__":
    main()
