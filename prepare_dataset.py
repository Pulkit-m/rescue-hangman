# %% [markdown]
# # Data Preparation
# * This file contains utilities like: 
#     - Train test split
#     - dataset preparation for training Language model

# %%
import pandas as pd 
import numpy as np 
import os, sys, time, yaml
import random
print("Time Last Script Run: " + time.asctime())

import string 
import itertools
from tqdm import tqdm
ascii_lowercase = string.ascii_lowercase

# %% [markdown]
# # Train Test Split

# %%
def TrainTestSplit(raw_source, train_target, test_target): 
    # reading all words from the dictionary. 
    dictionaryFile = open(raw_source) 
    wordList = list(map(lambda x : x.strip() ,dictionaryFile.readlines()))
    lenDict = len(wordList) 
    dictionaryFile.close()
    print(str(lenDict) + " words found")

    # make the train and test folders to segregate data for training and validation. 
    os.makedirs(train_target, exist_ok=True) 
    os.makedirs(test_target, exist_ok=True) 

    # shuffling data from the source file.
    random.shuffle(wordList)
    train_size = int(0.8*lenDict)
    trainWordList = wordList[:train_size]
    testWordList = wordList[train_size:]

    # saving files: 
    train_path = f"{train_target}/train.txt" 
    with open(train_path, 'w') as f: 
        for word in trainWordList: 
            f.write(word + '\n') 
    test_path = f"{test_target}/test.txt" 
    with open(test_path, 'w') as f: 
        for word in testWordList: 
            f.write(word + '\n') 
                
    print(f"creating training and testing splits: \nTrain: {len(trainWordList)} \nTest: {len(testWordList)}")
    print(f"Saving train and test files to {train_path} & {test_path} respectively")
    return train_path, test_path

# %% [markdown]
# ___
# # Training Data Preparation

# %% [markdown]
# ### Final Modelling using three inputs: 
# 1. Input padded and one hot encoded word with blank spaces as '_'.
# 2. Input available characters from the english alphabet that are still availabe for gressing. 
# 3. Input wrong characters from the english alphabet that should not be used. 
# 
# Target: correctly missing characters. 

# %%
def allUniqCombinations(word, drop_combinations = 0):
    uniq_chars = np.unique(list(word)).tolist()
    num_uniq_chars = len(uniq_chars)
    uniq_combos = []
    for r in range(1,num_uniq_chars+1):
        combinations = list(itertools.combinations(uniq_chars,r)) 
        uniq_combos = uniq_combos + combinations

    N = len(uniq_combos) 
    drop = int(N*drop_combinations)
    combos = random.sample(uniq_combos,k=N-drop) 
    return combos


def wordDecay(word,chars_to_remove,num_wrong_guesses = 6): 
    original_word = word
    for ch in chars_to_remove: 
        word = word.replace(ch,'_') 

    alphabet = string.ascii_lowercase
    for ch in list(word):
        alphabet = alphabet.replace(ch,'')  

    num_wrong_guesses = random.randint(0,min(num_wrong_guesses,len(alphabet)//3))
    wrong_guesses = ''.join(list(filter(                            
                    lambda x : x not in chars_to_remove,            
                    random.sample(alphabet,k=num_wrong_guesses))))  

    for ch in list(wrong_guesses): 
        alphabet = alphabet.replace(ch,'')

    # word, chars_to_remove, alphabet, wrong_guesses
    return f"{original_word},{word},{''.join(chars_to_remove)},{alphabet},{wrong_guesses}"


def prepareDataset(source_path,save_path, drop_combinations = 0):
    with open(source_path,'r') as s: 
        wordList = list(map(lambda x : x.strip(), s.readlines()))
    
    file = open(save_path,'w') 
    for word in tqdm(wordList):
        uniq_removal_combinations = allUniqCombinations(word, drop_combinations) 
        dataset_word = []
        for combo in uniq_removal_combinations:
            dataset_word.append(wordDecay(word,combo)) 

        to_write = '\n'.join(dataset_word) 
        file.write(to_write) 
        file.write('\n')

    file.close()
    print("Loading File from storage") 
    df = pd.read_csv(save_path,names=["word","input","target","available","missed"])
    print(f"{df.shape[0]} records found in training dataset")
    print("Shuffling data")
    df = df.sample(frac=1) 
    df.to_csv(save_path[:-4] + ".csv",index=False)
    print("Saving data as a csv file.") 




# %%
if __name__=="__main__":      
    with open('./config.yaml', 'r') as f: 
        config = yaml.safe_load(f) 

    source_train_path, source_test_path = TrainTestSplit(config['data']['raw'],config['data']['train_target'], config['data']['test_target']) 
    drop_data_ratio = config['data']['drop_data_ratio'] # to reduce size of training data  

    save_train_path = os.path.join(config['data']['train_target'],'train_io.txt') # pass in a txt extension, will save both csv and txt
    prepareDataset(source_path=source_train_path,save_path=save_train_path, drop_combinations=drop_data_ratio) 

    save_test_path = os.path.join(config['data']['test_target'],'test_io.txt') # pass in a txt extension, will save both csv and txt
    prepareDataset(source_path=source_test_path,save_path=save_test_path, drop_combinations=drop_data_ratio) 

# %%


# %%



