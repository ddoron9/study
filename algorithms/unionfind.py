

class QuickUnion:
    def __init__(self, n):
        self.ids = [i for i in range(n)]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.ids[root_a] = root_b

    def find(self, a):
        parent = self.ids[a]
        while self.ids[parent] != parent:
            parent = self.ids[parent]
        return parent


if __name__=="__main__":
    uf = QuickUnion(10)
    uf.union(1,2)
    print(uf.ids)
    uf.union(4,6)
    print(uf.ids)
    uf.union(6,3)
    print(uf.ids)
    uf.union(2,4)
    print(uf.ids)