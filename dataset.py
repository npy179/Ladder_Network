#!/usr/bin/python
from __future__ import print_function
import numpy as np
import time

class Label_Dataset(object):

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
            rest_number = start + self.batch_size - _num_rows
            rest_dataset = self.dataset[start:_num_rows, :-1]
            rest_label = self.dataset[start:_num_rows, [-1]]
            append_index = np.random.permutation(_num_rows)[:rest_number]
            append_dataset = self.dataset[append_index, :-1]
            append_label = self.dataset[append_index, [-1]].reshape((-1, 1))
            data_batch = np.concatenate((rest_dataset, append_dataset), axis=0)
            label_batch = np.concatenate((rest_label, append_label), axis=0)

            self._batch_index = 0

        else:
            data_batch = self.dataset[start:end, :-1]
            label_batch = self.dataset[start:end, [-1]]

        return data_batch, label_batch

    def test_next_batch(self):
        _num_rows, _ = self.dataset.shape
        start = self._batch_index * self.batch_size
        end = (self._batch_index + 1) * self.batch_size

        self._batch_index += 1

        if end > _num_rows:
            end = _num_rows

        data_batch = self.dataset[start:end, :-1]
        label_batch = self.dataset[start:end, [-1]]

        return data_batch, label_batch


class Ulabel_Dataset(object):

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
            rest_number = start + self.batch_size - _num_rows
            rest_dataset = self.dataset[start:_num_rows, :]
            append_index = np.random.permutation(_num_rows)[:rest_number]
            append_dataset = self.dataset[append_index, :]
            data_batch = np.concatenate((rest_dataset, append_dataset), axis=0)
            self._batch_index = 0
        else:
            data_batch = self.dataset[start:end, :]

        return data_batch

class SemiDataset(object):
    def __init__(self, label_data, ulabel_data, batch_size):
        self.label_data = Label_Dataset(label_data, batch_size)
        self.ulabel_data = Ulabel_Dataset(ulabel_data, batch_size)

    def next_batch(self):
        label_data_batch, label_label_batch  = self.label_data.next_batch()
        ulabel_data_batch = self.ulabel_data.next_batch()
        data_batch = np.vstack((label_data_batch, ulabel_data_batch))
        return data_batch, label_label_batch

def main():

    label_dataset = np.load("label_dataset_sample.npy")
    unlabel_dataset = np.load("unlabel_dataset_sample.npy")
    start = time.time()
    ds = SemiDataset(label_dataset, unlabel_dataset, 5)
    run = 0
    while run < 100000:
        db, llb  = ds.next_batch()
        print(db.shape, llb.shape)
        run += 1

    time_used = time.time() - start

    print("time is : ", time_used)

if __name__=="__main__":
    main()
