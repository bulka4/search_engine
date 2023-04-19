"""
This file creates an excel file 'data/sentences_tables.xlsx' based on another excel file 'data/tables_key_sentences.xlsx'.
File 'data/sentences_tables.xlsx' contains sample sentenctes with matching tables.
In order to create that file we take key sentences with matching tables from the file tables_key_sentences,
create all possible combinations from those key sentences and take no more then 100 random samples
from those combinations (because sometimes number of those combinations is too big)
"""

import pandas as pd
import random
from tqdm import tqdm
import itertools

def remove_duplicates_from_list(lst):
    return list(dict.fromkeys(lst))

tables_key_sentences = pd.read_excel('data/tables_key_sentences.xlsx')
sentences_tables = pd.read_excel('data/sentences_tables.xlsx')

for index, (key_sentences, table) in tqdm(tables_key_sentences.iterrows()):
    key_sentences = key_sentences.split(', ')    
    sentences_combined = key_sentences.copy()

    # creating all possible combinations from sentences in key_sentences such that there are no duplicated words
    for subset_length in range(2, len(key_sentences) + 1):
        combinations = list(itertools.combinations(key_sentences, subset_length))
        combinations_without_duplicates = [' '.join(remove_duplicates_from_list(' '.join(combinations).split())) for combinations in combinations]

        sentences_combined += combinations_without_duplicates

    sentences_combined = remove_duplicates_from_list(sentences_combined)
    # take no more then 100 random samples form sentences_combined
    random.shuffle(sentences_combined)
    sentences_combined = sentences_combined[:100]
    
    for sentence in sentences_combined:
        new_row = pd.DataFrame({'Sentence': [sentence], 'Table': [table]})
        sentences_tables = pd.concat((sentences_tables, new_row))
    
sentences_tables.to_excel('data/sentences_tables.xlsx', index = False)