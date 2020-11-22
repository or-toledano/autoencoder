"""Seperate file for hyper-tuned functions because of multiprocessing limitation
"""
from train import run_trainer


def batch_partial(batch_size):
    run_trainer(epochs=10, batch_size=batch_size, half_depth=5, loss='l2')


def loss_partial(loss_type):
    run_trainer(epochs=10, batch_size=144, half_depth=5, loss=loss_type)


def epochs_partial(epoch_num):
    run_trainer(epochs=epoch_num, batch_size=144, half_depth=5, loss='l2')