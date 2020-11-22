import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor, nn, optim
from torch.nn import functional as fun
from torchvision.utils import save_image
from typing import List, Tuple
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from timewrap import timing
from pathlib import Path
import argparse

BATCHES_PER_EPOCH = 40  # TODO: for GPU, please increase so batch_size*BATCHES_PER_EPOCH ~= 200,000
PLOT_ALL = True
PRINT_ITER = 10
PREFIX = f''
RESULTS_FOLDER = 'results'

# Less likely you would need to change these:

IMG_DIM = 128
LATENT_DIM = 256
TRAIN_RATIO = .95
TEST_RATIO = .05
IMG_FOLDER = 'CelebA'
VISUALIZE_IMAGE_NUM = 16
CHANNELS = 3

np.random.seed(42)
torch.manual_seed(42)
GPU = torch.cuda.is_available()


def load_train_test(batch_size) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    global BATCHES_PER_EPOCH
    data = datasets.ImageFolder(IMG_FOLDER, transform=transforms.Compose([transforms.CenterCrop((128, 128)),
                                                                          transforms.ToTensor()]))
    n_data = len(data)
    BATCHES_PER_EPOCH = int(np.floor(n_data/batch_size))
    indices = list(range(n_data))
    np.random.shuffle(indices)
    indices = indices[:batch_size * BATCHES_PER_EPOCH]
    split = int(np.floor(TEST_RATIO * len(indices)))
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=GPU,
                                               pin_memory=GPU)
    test_loader = torch.utils.data.DataLoader(data, sampler=test_sampler, batch_size=batch_size, num_workers=GPU,
                                              pin_memory=GPU)
    return train_loader, test_loader, len(train_idx), len(test_idx)


class AutoEncoder(nn.Module):
    latent_dim = LATENT_DIM

    def __init__(self, half_depth) -> None:
        super(AutoEncoder, self).__init__()

        modules = list()
        first_dim = IMG_DIM >> 1
        dims = [first_dim << i for i in range(half_depth)]

        in_channels = CHANNELS
        dim: int = 0
        for dim in dims:
            modules.extend([nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(dim),
                            nn.LeakyReLU()])
            in_channels = dim
        # FC layer (full sized kernel convolution, acts the same)
        modules.extend([nn.Conv2d(dim, self.latent_dim, kernel_size=IMG_DIM >> half_depth, stride=1, padding=0),
                        nn.BatchNorm2d(self.latent_dim),
                        nn.LeakyReLU()])

        self.encoder = nn.Sequential(*modules)

        dims.reverse()

        modules = [nn.ConvTranspose2d(self.latent_dim, dim, kernel_size=IMG_DIM >> half_depth, stride=1, padding=0),
                   nn.BatchNorm2d(dim), nn.LeakyReLU()]

        dims.append(CHANNELS)
        for dim in dims[1:]:
            modules.extend([
                nn.ConvTranspose2d(in_channels, out_channels=dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(dim), nn.LeakyReLU()])
            in_channels = dim

        self.decoder = nn.Sequential(*modules)

    def forward(self, source: Tensor) -> Tensor:
        encoded = self.encoder(source)
        decoded = self.decoder(encoded)
        return decoded


class Trainer:
    def __init__(self, device, epochs, batch_size, half_depth, loss: str):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.s_loss = loss
        self.loss_function = fun.l1_loss if loss == 'l1' else fun.mse_loss
        self.half_depth = half_depth
        self.train_loader, self.test_loader, self.total_train, self.total_test = load_train_test(batch_size)
        self.train_losses: List[float] = list()
        self.test_losses: List[float] = list()
        self.model = AutoEncoder(half_depth).to(device)
        self.optimizer = optim.Adam(self.model.parameters())

    def __str__(self):
        return f'epochs_{self.epochs}_batch_{self.batch_size}_loss_{self.s_loss}_enc_depth{self.half_depth}'

    def plot_train_and_test_losses(self, epoch=99):
        plt.plot(self.train_losses)
        plt.plot(self.test_losses)
        plt.legend(['train', 'test'])
        plt.title('Results')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'{RESULTS_FOLDER}/epoch_error_graph_{self}_epoch_{epoch}.png')
        if PLOT_ALL:
            plt.show()

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        batch_idx = 0
        for data in self.train_loader:
            data = data[0].to(self.device)
            self.optimizer.zero_grad()
            recon = self.model(data)
            loss = self.loss_function(recon, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % PRINT_ITER == 0:
                print(f'Train epoch {epoch} {batch_idx * len(data)}/{self.total_train} '
                      f'({100. * batch_idx / self.total_train:.2f}%) average batch loss: {loss.item() / len(data):.5f}')
            batch_idx += 1

        train_loss /= self.total_train
        self.train_losses.append(train_loss)
        print(f'***** Epoch {epoch} average train loss is {train_loss:.5f}')

    def test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        i = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data[0].to(self.device)
                recon = self.model(data)
                loss = self.loss_function(recon, data).item()
                test_loss += loss
                if i == 0:
                    n_vis = min(VISUALIZE_IMAGE_NUM, data.size(0))
                    pairs = torch.cat([data[:n_vis], recon.view(-1, CHANNELS, IMG_DIM, IMG_DIM)[:n_vis]])
                    save_image(pairs.cpu(),
                               f'{RESULTS_FOLDER}/{PREFIX}bpe_{BATCHES_PER_EPOCH}_{self}_epoch_{epoch}.png', nrow=n_vis)
                    self.plot_train_and_test_losses(epoch)
                if i % PRINT_ITER == 0:
                    print(f'Test epoch {epoch} {i * len(data)}/{self.total_test} '
                          f'({100. * i / self.total_test:.2f}%) average batch loss: {loss / len(data):.5f}')
                i += 1

        test_loss /= self.total_test
        self.test_losses.append(test_loss)
        print(f'***** Epoch {epoch} average test loss is {test_loss:.5f}')


@timing
def run_trainer(epochs=2, batch_size=144, half_depth=5, loss='l2'):
    assert 1 <= half_depth <= 6, "Bad number of layers. Input size is reduced by 2**layers"
    Path(RESULTS_FOLDER).mkdir(exist_ok=True)
    device = torch.device("cuda" if GPU else "cpu")
    trainer = Trainer(device, epochs, batch_size, half_depth, loss)

    for epoch in range(0, trainer.epochs):
        trainer.train(epoch)
        trainer.test_epoch(epoch)

    trainer.plot_train_and_test_losses()

def get_args():
    parser = argparse.ArgumentParser(description='Trains and tests a autoencoder.')
    parser.add_argument('-bs', '--batch_size', type=int, default=144)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-h', '--half_depth', type=int, default=5)
    parser.add_argument('-l', '--loss', type=str, default='l1')


    return parser.parse_args()
def main():
    args = get_args()
    run_trainer(epochs=args.epochs, batch_size=args.batch_size, half_depth=args.half_depth, loss=args.loss)

    # loss_list = ['l1', 'l2']
    # epochs_list = [10, 100, 400]
    # half_depth_list = [2, 3, 4, 5]
    # batch_size_list = [10, 100, 1000, 10000]
    # for e in epochs_list:
    #     for l in loss_list:
    #         for b in batch_size_list:
    #             for d in half_depth_list:
    #                 run_trainer(epochs=e, batch_size=b, half_depth=d, loss=l)


if __name__ == "__main__":
    main()
