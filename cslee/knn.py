import numpy as np


def ecludDist(inA, inB):
    distance = np.sqrt(np.sum((inA-inB)**2, 1))  # 1번 축에 따라
    return distance

# point가 knn할 값 group이 이미 분류되어 있는 기존값


def dist(point, group):
    # argsort는 인덱스를 거리가 작은 순서대로 보냄
    dist_lst = list(np.argsort(list(ecludDist(point, group))))
    return dist_lst


def Read():
    dataset = np.loadtxt("./datingTestSet2.txt", delimiter='\t')
    trData = dataset[:900, :3]
    trLabels = list(dataset[:900, 3])
    testData = dataset[900:, :3]
    # 필요없음
    #testLabels = list(dataset[900:, 3])
    return testData, trData, trLabels


def KNN(test, train, labels, k):
    tested = []
    for i in test:
        # train 되어 있는 것중에서 k 개의 근접 이웃 거리인 train data의 인덱스  each_labels 로
        each_labels = dist(i, train)[:k]
        dict = {}
        # 중복 제거 후 키 값 0 초기화
        print('set(labels)= ', set(labels))
        for j in set(labels):
            dict[j] = 0
        # train label 에 해당하는 인덱스 값인 idx 에 해당하는 key 값 count
        for idx in each_labels:
            dict[labels[idx]] += 1

        # 가장 많이 나온 라벨인 max_l 와 그 횟수 max
        for j in set(labels):
            max = 0
            if max < dict[j]:
                max, max_l = dict[j], j
        # 분류를 완료한 라벨을 추가
        tested.append(max_l)
    return tested


a, b, c = Read()
print(KNN(a, b, c, 30))
