from Member import Member
from Cluster import Cluster
import numpy as np
from collections import defaultdict

class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._cluster = [Cluster() for _ in range(num_clusters)]
        self._E = [] # list of centroids
        self._S = 0 # overall similarity
    
    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            """
            Convert a sparse representation of a document by tf-idf to dense form
            by mapping its word to a new vector of vocabulary size.
            Word without any appearance in the doc get the value 0.0
            """
            r_d = [0.0 for _ in range (vocab_size)]
            indices_tf_idfs = sparse_r_d.split()
            for index_tf_idf in indices_tf_idfs:
                index = int(index_tf_idf.split(":")[0])
                tfidf = float(index_tf_idf.split(":")[1])
                r_d[index] = tfidf
            return np.array(r_d)
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open("../datasets/20news-bydate/words_idfs.txt") as f:
            vocab_size = len(f.read().splitlines())
        
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size = vocab_size)

            self._data.append(Member(r_d=r_d, label = label, doc_id = doc_id))
        
        def random_init(self, seed_value):

        def compute_similarity(self, member, centroid):

        def select_cluster_for(self, member):
            best_fit_cluster = None
            max_similarity = -1
            for cluster in self._clusters:
                similarity = self.compute_similarity(member, cluster)
                if similarity > max_similarity:
                    best_fit_cluster = cluster
                    max_similarity = similarity
            best_fit_cluster.add_member(member)
            return max_similarity
        def stopping_condition(self, criterion, threshold):
            criteria = ["centroid", "similarity", "max_iters"]
            assert criterion in criteria
            if criterion == "max_iters":
                if self._iteration >= threshold:
                    return True
                else:
                    return False
            elif criterion = "centroid":
                # the number of changed centroids are not significant
                

        def run(self, seed_value, criterion, threshold):

        def compute_purity(self):

        def compute_NMI(self):




