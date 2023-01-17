import os
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
from collections import defaultdict

def get_tf_idf(data_path):
    with open ("../datasets/20news-bydate/words_idfs.txt") as f:
        word_idfs = [(line.split("<fff>")[0], float(line.split("<fff>")[1]))  
                    for line in f.read().splitlines()]

        idfs = dict(word_idfs)
        word_IDs = dict([(word, index) for index, (word, idf) in enumerate(word_idfs)])
    
    with open (data_path) as f:
        documents = [
            (int(line.split("<fff>")[0]),
            int(line.split("<fff>")[1]),
            line.split("<fff>")[2])
            for line in f.read().splitlines()]
        total_doc_num = len(documents)
        
    data_tf_idf = []
    for i, document in enumerate(documents):
        if i % 100 == 0:
            print("Processing {i}-th/{total_doc_num} document".format(i = i, total_doc_num = total_doc_num))
        # unpack document
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])

        words_tf_idf = []
        sum_squares = 0.0
        
        for word in word_set:
            term_freq = words.count(word)

            tf_idf_value = term_freq * 1. / max_term_freq * idfs[word]
            word_idfs.append((word_IDs[word], tf_idf_value))
            sum_squares = sum_squares + tf_idf_value ** 2
        
        words_tfidfs_normalized = [str(index)+":"+ str(tf_idf_value / np.sqrt(sum_squares))
                                for index, tf_idf_value in word_idfs]
        spare_rep = " ".join(words_tfidfs_normalized)
        data_tf_idf.append((label, doc_id, spare_rep))

    with open('../datasets/20news-bydate/words_tf_idfs.txt', "w") as f:
        f.write("\n".join([str(label) + "<fff>" + str(doc_id) + "<fff>" + spare_rep
                for label, doc_id, spare_rep in data_tf_idf]))



path = "/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/"
train_dir = path + "20news-train-processed.txt"
# with open (train_dir) as f:
#     lines = f.read().splitlines()
#     print(len(lines))
get_tf_idf(train_dir)