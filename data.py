import torch

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    Saves the total number of queries to datapoints.

    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/4
    """

    def __init__(
            self, *tensors, 
            batch_size=32, shuffle=False, keep_last=True,
            generator=None, device=None):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param keep_last: keep last batch if not full

        :returns: A FastTensorDataLoader.
        """
        
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_queries = 0
        self.keep_last = keep_last

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if n_batches == 0:
            self.keep_last = True
        if remainder > 0 and self.keep_last:
            n_batches += 1
        self.n_batches = n_batches

        self.generator = generator
        self.device = device
        if device is not None:
            self.tensors = [t.to(device) for t in self.tensors]

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(
                self.dataset_len, generator=self.generator)
            if self.device is not None:
                self.indices = self.indices.to(self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if not self.keep_last and self.i + self.batch_size > self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices)
                          for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size]
                          for t in self.tensors)
        self.i += self.batch_size
        self.num_queries += batch[0].shape[0]
        return batch

    def __len__(self):
        return self.n_batches