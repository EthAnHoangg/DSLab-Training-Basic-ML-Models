from Member import Member
from Cluster import Cluster
import numpy as np
from collections import defaultdict


class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._cluster = [Cluster() for _ in range(num_clusters)]
        self._E = []  # list of centroids
        self._S = 0  # overall similarity

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            """
            Convert a sparse representation of a document by tf-idf to dense form
            by mapping its word to a new vector of vocabulary size.
            Word without any appearance in the doc get the value 0.0
            """
            r_d = [0.0 for _ in range(vocab_size)]
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
            r_d = sparse_to_dense(
                sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))
      
    def random_init(self, seed_value):
        np.random.seed(seed_value)
        total_docs = (len(self._data))
        for i in range (self._num_clusters):
            centroid_idx = np.random.randint(0, total_docs)
            self._E.append(self._data[centroid_idx])
            self._cluster[i]._centroid = self._data[centroid_idx]._r_d
            
    def compute_similarity(self, member, centroid):
        # euclidean distance, Kmeans algorithm try to reduce squared error
        # print(np.sqrt(np.sum((member._r_d - centroid._r_d) ** 2)))
        return np.sqrt(np.sum((member._r_d - centroid) ** 2))

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._cluster:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_members(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        # list of r_d of members in considering cluster
        aver_r_d = np.mean(member_r_ds, axis=0)
        # calcualter the mean of all members
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([value/sqrt_sum_sqr for value in aver_r_d])
        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        criteria = ["centroid", "similarity", "max_iters"]
        assert criterion in criteria
        if criterion == "max_iters":
            return True if self._iteration >= threshold else False
        elif criterion == "centroid":
            # the number of changed centroids are not significant
            E_new = [list(cluster._centroid) for cluster in self._cluster]
            E_new_minus_E = [
                centroid for centroid in E_new if centroid not in self._E]

            self._E = E_new
            return True if len(E_new_minus_E) <= threshold else False
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            return True if new_S_minus_S <= threshold else False

    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value=seed_value)
        self._iteration = 0
        # update cluster continuously
        while True:
            print("=========== Processing {}-th epoch =============".format(self._iteration))
            for cluster in self._cluster:
                cluster.reset_members()  # empty the members of cluster but not the centroid
            self._new_S = 0
            print("Selecting cluster for all members")
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            print("Updated all members")
            for i, cluster in enumerate(self._cluster):
                # print(i) #for debugging
                self.update_centroid_of(cluster)
            print("Updated centroids")
            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break

    def compute_purity(self):
        majority_sum = 0
        for cluster in self._cluster:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label)
                            for label in range(20)])
            # 20 is the range of label corresponding to 20 categories
            majority_sum += max_count
        return majority_sum * 1. / len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._cluster:
            wk = len(cluster._members) * 1.
            H_omega += -wk/N * np.log10(wk/N)
            member_labels = [member._label for member in cluster._members]

            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += wk_cj/N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += -cj/N * np.log10(cj/N)
        return I_value * 2./(H_omega + H_C)
