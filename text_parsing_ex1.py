#!/usr/bin/env python

# Inspired by https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html
import collections
import math
import numpy as np
import matplotlib.pyplot as plt

# Configuration
batch_size = 20
# Dimension of the embedding vector. Two too small to get
# any meaningful embeddings, but let's make it 2 for simple visualization
embedding_size = 2
num_sampled = 15    # Number of negative examples to sample.

# Sample sentences
sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]

# sentences to words and count
words = " ".join(sentences).split()
print ("========================================")
print ("words: ", words)
#=> words:  ['the', 'quick', 'brown', 'fox', 'jumped'...
count = collections.Counter(words).most_common()
print ("========================================")
print ("Word count(Top 5): ", count[:5])
#=> Word count(Top 5):  [('cats', 10), ('dogs', 6), ('and', 5), ('are', 4), ('love', 3)]

# Build dictionaries
rdic = [i[0] for i in count] #reverse dic, idx -> word
print ("========================================")
print (rdic)
#=> ['cats', 'dogs', 'and', 'are', 'love', 'the', ....]

dic = {w: i for i, w in enumerate(rdic)} #dic, word -> id
print ("========================================")
print (dic)
#=> {..., 'love': 4,..., 'cats': 0,..., 'and': 2,.. 'are': 3,... 'dogs': 1...}

voc_size = len(dic)
print ("========================================")
print('voc_size (=len(dic)): ', voc_size)

# Make indexed word data
data = [dic[word] for word in words]
print ("========================================")
print('Sample data', data[:10], [rdic[t] for t in data[:10]])


