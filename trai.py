from __future__ import print_function

from torch import Tensor
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms

latent_dim = 256

PRINT_ITER = 10
TEST_RATIO = .05
IMG_FOLDER = 'CelebA'


def load_split_train_test(batch_size, cpu) -> Tuple[DataLoader, DataLoader]:
    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.ToTensor(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
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


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim  # TODO

        modules = []
        hidden_dims = [13, 843, 323, 7, 13]

        in_channels = 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class Trainer:
    def __init__(self, device, epochs, batch_size, cpu: bool):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.cpu = cpu
        self.train_loader, self.test_loader = load_split_train_test(batch_size, cpu)
        self.train_losses: List[float] = list()
        self.test_losses: List[float] = list()
        self.model = AutoEncoder().to(device)
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
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.test_loader.dataset),
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
                                            recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
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
            sam = trainer.model.sample(64, trainer.device).cpu()
            save_image(sam.view(64, 3, 128, 128),
                       'results/sample_' + str(epoch) + '.png')

    trainer.plot_train_losses()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='force CPU usage')
    args = parser.parse_args()
    args.cpu = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu" if args.cpu else "cuda")
    torch.manual_seed(42)

    trainer = Trainer(device, epochs=3, batch_size=100, cpu=args.cpu)
    run_trainer(trainer)


if __name__ == "__main__":
    main()
