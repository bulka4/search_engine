import json
import re
import pandas as pd
import numpy as np
import os

from tensorflow.keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class Data_preparation():
    def __init__(self, 
                 # embedding_file_path = r'data\glove.840B.300d.txt', 
                 # vector_dimension = 300, 
                 excel_file_path = 'data/sentences_tables.xlsx',
                 tokenizer = Tokenizer()
                ):
        """
        embedding_file_path argument is path to a word embedding file which is a text file of a format:
        'word1 x1 x2 x3
        word2 x1 x2 x3'
        where x1, x2, x3 are values from a vector representation of a wrod.
        
        vector_dimension argument is a length of a vector which we assign to each word.
        
        excel_file argument is a path to an excel file with sentences and matching tables
        """
        
        self.excel_file_path = excel_file_path
        
        self.sentences_tables = pd.read_excel(excel_file_path)
        self.sentences = self.sentences_tables.Sentence.values
        self.tables = self.sentences_tables.Table.values
        
        self.tokenizer = tokenizer
        # self.tokenizer.fit_on_texts(self.sentences)
        
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')


    def create_embedding_matrix(self, vector_dimension, embedding_file_path = None):
        """
        A function to create the embedding matrix. This is a matrix where each row is a vector representing a word.
        To create that matrix we use word embedding file.
        embedding_matrix[row_number] is a vector representation for a word = list(self.tokenizer.word_index.keys())[row_number - 1]
        """
        if ~hasattr(self, 'clean_sentences'):
            self.clean_sentences = [clean_text(sentence) for sentence in self.sentences]
            self.tokenizer.fit_on_texts(self.clean_sentences)
        
        if os.path.isfile('data\embedding_matrix.csv'):
            embedding_matrix = pd.read_csv('data\embedding_matrix.csv')
        else: 
            if os.path.isfile('data\embeddings_index.json'):
                with open('data\embeddings_index.json', 'r') as file:
                    embeddings_index = json.load(file)
            else:
                embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(embedding_file_path, errors = 'ignore'))
                with open('data\embeddings_index.json', 'w') as file:
                    json.dumb(embeddings_index, file)

            embedding_matrix = np.zeros((len(self.tokenizer.word_counts) + 1, vector_dimension))
            for word, index in self.tokenizer.word_index.items():
                if index > len(self.tokenizer.word_counts):
                    break
                else:
                    try:
                        embedding_matrix[index] = embeddings_index[word]
                    except:
                        continue

        return embedding_matrix


    def clean_text(string: str, 
                   punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
                   stop_words = stopwords.words('english'),
                   porter = PorterStemmer()
                  ):
        """
        A method to clean text 
        """
        # Removing the punctuations
        for x in string.lower(): 
            if x in punctuations: 
                string = string.replace(x, "") 

        # Converting the text to lower
        string = string.lower()

        # Removing stop words
        string = ' '.join([word for word in string.split() if word not in stop_words])

        # stemming words. That means changing word to its basic format, for example
        # words 'fishing', 'fished', 'fischer' will be changed into a word 'fisch'
        string = ' '.join([porter.stem(word) for word in string.split()])

        # Cleaning the whitespaces
        string = re.sub(r'\s+', ' ', string).strip()

        return string


    def vectorize(self, vector_length):
        """
        This function takes a sentence which is a string and uses a self.tokenizer to change it into a vector of length vector_length.
        """
        sequence = self.tokenizer.texts_to_sequences(self.sentences)
        return pad_sequences(sequence, maxlen = vector_length).flatten()


    def create_train_data(self):
        if ~hasattr(self, 'clean_sentences'):
            self.clean_sentences = [clean_text(sentence) for sentence in self.sentences]
            self.tokenizer.fit_on_texts(self.clean_sentences)
            
        y_train_str = self.tables
        all_unique_tables = np.unique(y_train_str)
        y_train_num = [np.where(all_unique_tables == table)[0][0] for table in y_train_str]

        max_len = np.max([len(sentence.split()) for sentence in self.clean_sentences])
        x_train = np.array([vectorize(sentence, max_len, tokenizer) for sentence in self.clean_sentences])

        return x_train, y_train_str, y_train_num