from Kmeans import Kmeans
from Cluster import Cluster
from Member import Member

model  = Kmeans(num_clusters = 20)
model.load_data("../datasets/20news-bydate/data_tf_idf.txt")
print(model._data[:10])