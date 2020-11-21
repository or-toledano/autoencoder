from hypertuned_funcs import batch_partial

from multiprocessing import Pool
import tqdm

POOL = Pool(processes=7)
GPU = False


def tune_batch_size():
    sizes = [i ** 2 for i in range(8, 14)]
    for _ in tqdm.tqdm(POOL.map(batch_partial, sizes), total=len(sizes)):
        pass


def main():
    tune_batch_size()


if __name__ == "__main__":
    main()
