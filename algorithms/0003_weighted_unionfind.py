

class WeightedQuickUnion:
    def __init__(self, n):
        self.ids = [i for i in range(n)]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.ids[root_a] = root_b

    def find(self, a):
        while a != self.ids[a]:
            self.ids[a] = self.ids[self.ids[a]]
            a = self.ids[a]
        return a


if __name__=="__main__":
    uf = WeightedQuickUnion(10)
    uf.union(1,2)
    print(uf.ids)
    uf.union(4,6)
    print(uf.ids)
    uf.union(6,3)
    print(uf.ids)
    uf.union(2,4)
    print(uf.ids)
