#!/usr/bin/python
from __future__ import print_function
import numpy as np
import time

class Dataset(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
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

        data_batch = self.dataset[start:end, :-1]
        label_batch = self.dataset[start:end, [-1]]
        return data_batch, label_batch

class SemiDataset(Dataset):
    def __init__(self, label_data, ulabel_data, batch_size):
        self.label_data = Dataset(label_data, batch_size)
        self.ulabel_data = Dataset(ulabel_data, batch_size)

    def next_batch(self):
        label_data_batch, label_label_batch  = self.label_data.next_batch()
        ulabel_data_batch = self.ulabel_data.next_batch()

        return label_data_batch, label_label_batch, ulabel_data_batch

def main():

    label_dataset = np.load("label_dataset_sample.npy")
    unlabel_dataset = np.load("unlabel_dataset_sample.npy")
    start = time.time()
    ds = SemiDataset(label_dataset, unlabel_dataset, 2)
    run = 0
    while run < 10:
        ldb, llb, udb  = ds.next_batch()
        print(llb.shape)
        run += 1

    time_used = time.time() - start

    print("time is : ", time_used)

if __name__=="__main__":
    main()
