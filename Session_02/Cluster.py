class Cluster:
    def __init__(self):
        self.centroids = None
        self.member = []
    
    def reset_member(self):
        self.member = []
    def add_member(self, member):
        self.member.append(member)
        