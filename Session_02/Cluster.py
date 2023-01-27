class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
    
    def reset_members(self):
        self._members = []
    def add_members(self, member):
        self._members.append(member)
        