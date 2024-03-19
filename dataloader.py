# %%
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pandas as pd
import string

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
        
        # Convert target word to a one-d vector target
        def encode_word_to_vec(word):
            target = np.zeros(self.vocab_size) 
            if type(word)!=str:
                return target
            for ch in word:
                if ch != '_' and ch != '<pad>': 
                    index = self.char_to_index[ch]
                    target[index] = 1
            return target

        # Preprocess input and target columns
        input_data = data['input'].apply(one_hot_encoding).tolist()
        input_available = data['available'].apply(encode_word_to_vec).tolist() 
        target_data = data['target'].apply(encode_word_to_vec).tolist()

        # print(type(input_data), type(input_data[0]))

        return input_data, input_available, target_data

    def __iter__(self):
        # Load CSV file
        for chunk in pd.read_csv(self.csv_file, chunksize=self.chunk_size):
            # print(chunk.head())
            # Convert input and target columns to tensors
            input_data,input_available, target_data = self.preprocess_data(chunk)
            yield torch.tensor(input_data, dtype=torch.float32),\
                torch.tensor(input_available, dtype = torch.float32),\
                torch.tensor(target_data, dtype=torch.float32)

    # def fit(self):
    #     # Build character to index mapping
    #     self.char_to_index = {char: index for index, char in enumerate('abcdefghijklmnopqrstuvwxyz_')}
    #     self.char_to_index['<pad>'] = len(self.char_to_index)
            

class HangmanTextEmbeddedDataset(IterableDataset):
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
        def index_encoding(word):
            one_hot = np.zeros(self.seq_length)
            for i, ch in enumerate(word):
                index = self.char_to_index.get(ch)  # Get index from char_to_index dict
                one_hot[i] = index
            return one_hot
        
        # Convert target word to a one-d vector target
        def encode_word_to_vec(word):
            target = np.zeros(self.vocab_size) 
            if type(word)!=str:
                return target
            for ch in word:
                if ch != '_' and ch != '<pad>': 
                    index = self.char_to_index[ch]
                    target[index] = 1
            return target

        # Preprocess input and target columns
        input_data = data['input'].apply(index_encoding).tolist()
        input_available = data['available'].apply(encode_word_to_vec).tolist() 
        target_data = data['target'].apply(encode_word_to_vec).tolist()

        # print(type(input_data), type(input_data[0]))

        return input_data, input_available, target_data

    def __iter__(self):
        # Load CSV file
        for chunk in pd.read_csv(self.csv_file, chunksize=self.chunk_size):
            print(chunk.head())
            # Convert input and target columns to tensors
            input_data,input_available, target_data = self.preprocess_data(chunk)
            yield torch.tensor(input_data, dtype=torch.int),\
                torch.tensor(input_available, dtype = torch.int),\
                torch.tensor(target_data, dtype=torch.int)




# Example usage:
if __name__ == "__main__":
    dataset = HangmanTextDataset('./data/train/train_mini.csv', chunk_size=32)
    # dataset.fit()

    dataloader = DataLoader(dataset)
    print("Behold the training dataset: ")
    for batch_input, batch_available, batch_target in dataloader:
        # Train your model using batch_input and batch_target
        print("loading batches of chunk size 32: first only")
        print("input batch shape: ", torch.squeeze(batch_input).shape, "target_batch_shape: ",torch.squeeze(batch_target).shape)
        print("available batch shape: ", torch.squeeze(batch_available).shape)
        break

# %%



