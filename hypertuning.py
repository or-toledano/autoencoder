from hypertuned_funcs import batch_partial, loss_partial, epochs_partial

from multiprocessing import Pool
import tqdm

GPU = False


def tune_batch_size():
    sizes = [i ** 2 for i in range(8, 14)]
    with Pool(7) as pool:
        for _ in tqdm.tqdm(pool.map(batch_partial, sizes), total=len(sizes)):
            pass


def tune_loss_type():
    loss_list = ['l1', 'l2']
    with Pool(7) as pool:
        for _ in tqdm.tqdm(pool.map(loss_partial, loss_list), total=len(loss_list)):
            pass


def tune_epoch_num():
    epochs_list = [10, 40, 250]
    with Pool(7) as pool:
        for _ in tqdm.tqdm(pool.map(epochs_partial, epochs_list), total=len(epochs_list)):
            pass


def main():
    tune_batch_size()
    tune_loss_type()
    tune_epoch_num()


if __name__ == "__main__":
    main()
