# %%
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import string
import pandas as pd

class HangmanTextDataset(IterableDataset):
    def __init__(self, csv_file, vocab_size=28, seq_length=40, batch_size=1, chunk_size = 64):
        self.csv_file = csv_file
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.vocab = ['<pad>'] + list(string.ascii_lowercase) + ['_']
        self.char_to_index = {c:i for i,c in enumerate(self.vocab)}

    def preprocess_data(self, data):
        # Convert characters to one-hot encoding
        def one_hot_encoding(word):
            one_hot = np.zeros((self.seq_length, self.vocab_size))
            for i, ch in enumerate(word):
                index = self.char_to_index.get(ch)  # Get index from char_to_index dict
                one_hot[i][index] = 1
            return one_hot
        
        # Convert target word to multi-class classification target
        def encode_target_word(word):
            target = np.zeros(self.vocab_size) 
            for ch in word:
                if ch != '_' and ch != '<pad>':
                    index = self.char_to_index[ch]
                    target[index] = 1
            return target

        # Preprocess input and target columns
        input_data = data['input'].apply(one_hot_encoding).tolist()
        target_data = data['target'].apply(encode_target_word).tolist()

        return input_data, target_data

    def __iter__(self):
        # Load CSV file
        for chunk in pd.read_csv(self.csv_file, chunksize=self.chunk_size):
            # Convert input and target columns to tensors
            input_data, target_data = self.preprocess_data(chunk)
            yield torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)


# Example usage:
if __name__ == "__main__":
    dataset = HangmanTextDataset('./data/train/train_io.csv', chunk_size=8)
    # dataset.fit()

    dataloader = DataLoader(dataset)
    for batch_input, batch_target in dataloader:
        # Train your model using batch_input and batch_target
        print("loading batches: first only")
        print("input batch shape: ", torch.squeeze(batch_input).shape, "target_batch_shape: ",torch.squeeze(batch_target).shape)
        break

# pending performance issue: 
# <ipython-input-3-deb6d007de5b>:47: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting 
# the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ..\torch\csrc\utils\tensor_new.cpp:278.)
#   yield torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)

