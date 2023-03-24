# Recollecting data
## Build the vocab and collectin data from files

import numpy as np
import os
from collections import defaultdict
import re

def gen_data_and_vocab():
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = os.path.join(parent_path, newsgroup)
            files = [(filename, os.path.join(dir_path, filename)) for filename in os.listdir(dir_path)\
                     if os.path.isfile(os.path.join(dir_path, filename))]
            files.sort()
            label = group_id
            print("Processing: {}-{}".format(group_id, newsgroup))
            
            for filename, filepath in files:
                with open(filepath, errors= "ignore") as f:
                    text = f.read().lower()
                    words = re.split("\W+", text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content = " ".join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + "<fff>" + filename + "<fff>" + content)
        return data
    
    word_count = defaultdict(int)
    train_path = "/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/20news-bydate-train"
    test_path = "/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/20news-bydate-test"
    
    newsgroup_list = [newsgroup for newsgroup in os.listdir(train_path)]
    newsgroup_list.sort()
    
    train_data = collect_data_from(
        parent_path=train_path,
        newsgroup_list=newsgroup_list,
        word_count=word_count
    )
    print("All files in train set have already been processed!")
    test_data = collect_data_from(
        parent_path=test_path,
        newsgroup_list=newsgroup_list
    )
    print("All files in test set have already been processed!")
# ================= Gen vocab ============================
    vocab = [word for word, freq in zip(word_count.keys(), word_count.values()) if freq > 10]
    vocab.sort()
# =================================================================
    
    with open("/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/w2v/vocab-raw.txt", "w") as f:
        f.write("\n".join(vocab))
    with open("/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/w2v/20news-train-raw.txt", "w") as f:
        f.write("\n".join(train_data))
    with open("/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/w2v/20news-test-raw.txt", "w") as f:
        f.write("\n".join(test_data))


unknown_ID = 0
padding_ID = 1

def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2)
                      for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])
                     for line in f.read().splitlines()]

    MAX_DOC_LENGTH = 500
    encoded_data = []
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)

        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_ID))

        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))

        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>'
                            + str(sentence_length) + '<fff>' + ' '.join(encoded_text))

    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
 

    with open(dir_name + '/' + file_name, 'w') as f:
        f.write('\n'.join(encoded_data))
        
gen_data_and_vocab()
encode_data(data_path = "/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/w2v/20news-train-raw.txt",
            vocab_path = "/Users/anhoang/Documents/GitHub/DSLab-training-phase/datasets/20news-bydate/w2v/vocab-raw.txt")
