class WeightedUnionFind:
    def __init__(self, n):
        self.ids = [i for i in range(n)]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.ids[root_a] = max(root_a, root_b)
        self.ids[root_b] = max(root_a, root_b)

    def find(self, a):
        maximum = a
        while a != self.ids[a]:
            self.ids[a] = self.ids[self.ids[a]]
            a = self.ids[a]
            maximum = max(a, maximum)
        return maximum

    def connected(self, a, b):
        return self.find(a) == self.find(b)

class SmallestSuccessor(WeightedUnionFind):
    def __init__(self, n):
        super().__init__(n)
        self.n = n

    def delete(self, a):
        assert 0 <= a < self.n
        if a + 1 >= self.n:
            self.n -= 1
            return
        return self.union(self.ids[a], self.ids[a+1])

    def successor(self, a):
        assert 0 <= a < self.n

        while a + 1 >= self.n:
            a -= 1

        return self.ids[a+1]
