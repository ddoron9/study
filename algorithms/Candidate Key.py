# https://school.programmers.co.kr/learn/courses/30/lessons/42890

from itertools import combinations

def is_candidate(combination):
    global data
    candidate = [""] * len(data)
    # print(combination)
    for i, r in enumerate(data):
        for c in combination:
            candidate[i] += r[c]
    
    return len(candidate) == len(set(candidate))

data = []

def is_minimality(answer, comb):
    for a in answer:
        if set(a) == set(a) & set(comb):
            return False
    return True
            
def solution(relation):
    answer = []
    columns = [i for i in range(len(relation[0]))]
    global data
    data = relation
    for i in range(1, len(relation[0]) + 1):
        for comb in combinations(columns, i):
            if is_candidate(comb):
                if is_minimality(answer, comb):
                    answer.append(comb)
        
    return len(answer)
