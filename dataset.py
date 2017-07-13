#!/usr/bin/python
from __future__ import print_function
import numpy as np

class Dataset(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset #np.load(dataset, mmap_mode="r")
        self.batch_size = batch_size
        self._batch_index = 0

    def next_batch(self):
        _num_rows, _ = self.dataset.shape
        start = self._batch_index * self.batch_size

        end = (self._batch_index + 1) * self.batch_size

        self._batch_index += 1

        if end > _num_rows:
            end = _num_rows
            self._batch_index = 0

        batch = self.dataset[start:end, :]
        return batch

class SemiDataset(object):
    def __init__(self, label_data, label_label, ulabel_data, batch_size):
        self.label_data = Dataset(label_data, batch_size)
        self.label_label = Dataset(label_label, batch_size)
        self.ulabel_data = Dataset(ulabel_data, batch_size)

    def next_batch(self):
        label_data_batch = self.label_data.next_batch()
        label_label_batch = self.label_label.next_batch()
        ulabel_data_batch = self.ulabel_data.next_batch()
        semi_data_batch = np.vstack((label_data_batch, ulabel_data_batch))

        return semi_data_batch, label_label_batch

def main():
    sample_dataset_0 = np.arange(1251 * 2).reshape(1251, 2)
    np.save('sample_dataset_0', sample_dataset_0)

    sample_dataset_1 = np.arange(251 * 2).reshape(251, 2)
    np.save('sample_dataset_1', sample_dataset_1)

    sample_dataset_2 = np.arange(251 * 2).reshape(251, 2)
    np.save('sample_dataset_2', sample_dataset_2)

    sample_dataset_0 = np.load("sample_dataset_0.npy", mmap_mode="r")
    sample_dataset_1 = np.load("sample_dataset_1.npy", mmap_mode="r")
    sample_dataset_2 = np.load("sample_dataset_2.npy", mmap_mode="r")

    ds = SemiDataset(sample_dataset_1, sample_dataset_2, sample_dataset_0, 2)
    while 1:
        ds1, lb1 = ds.next_batch()
        print(ds1)


if __name__=="__main__":
    main()
