# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from matplotlib import pyplot as plt

from torchvision import transforms

try:
    from torchvision.datasets.mnist import MNIST
except ModuleNotFoundError:
    from tests.base.datasets import MNIST

PRINT_ITER = 10
TEST_RATIO = .05
IMG_FOLDER = 'CelebA'
TO_IMAGE = transforms.ToPILImage()


class AutoEncoder(pl.LightningModule):
    # def load_split_train_test(batch_size, cpu) -> Tuple[DataLoader, DataLoader]:
    #     train_transforms = transforms.Compose([transforms.Resize(128),
    #                                            transforms.ToTensor(),
    #                                            ])
    #     test_transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    #     train_data = datasets.ImageFolder(IMG_FOLDER, transform=train_transforms)
    #     test_data = datasets.ImageFolder(IMG_FOLDER, transform=test_transforms)
    #     num_train = len(train_data)
    #     indices = list(range(num_train))
    #     split = int(np.floor(TEST_RATIO * num_train))
    #     np.random.shuffle(indices)
    #     train_idx, test_idx = indices[split:], indices[:split]
    #     train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
    #     train_loader = torch.utils.data.DataLoader(train_data,
    #                                                sampler=train_sampler, batch_size=batch_size, num_workers=not cpu,
    #                                                pin_memory=not cpu)
    #     test_loader = torch.utils.data.DataLoader(test_data,
    #                                               sampler=test_sampler, batch_size=batch_size, num_workers=not cpu,
    #                                               pin_memory=not cpu)
    #     return train_loader, test_loader
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    model = AutoEncoder()

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)
    visualize(model, val_loader)


def visualize(model, val_loader):
    for batch in val_loader:
        original_imgs = batch[0].view(-1, 1, 1, 784)
        outputs = model(original_imgs)
        for i in range(len(outputs)):
            # Original Image
            plt.figure()
            plt.imshow(TO_IMAGE(outputs[i].view(-1, 1, 28, 28)).convert("RGB"))
            # Reconstructed
            plt.figure()
            plt.imshow(TO_IMAGE(original_imgs[i].view(-1, 1, 28, 28)).convert("RGB"))
            if i == 3:
                break
            break


if __name__ == '__main__':
    main()
