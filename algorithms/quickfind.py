

class QuickFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a < root_b:
            self.root[b] = a
        else:
            self.root[a] = b

    def find(self, a):
        parent = self.root[a]
        while self.root[parent] != parent:
            parent = self.root[parent]
        return parent


if __name__=="__main__":
    uf = QuickFind(10)
    uf.union(1,2)
    print(uf.root)
    uf.union(4,6)
    print(uf.root)
    uf.union(6,3)
    print(uf.root)
