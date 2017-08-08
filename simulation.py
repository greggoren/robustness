from copy import copy
import itertools
from itertools import imap
from math import log
import sys
import scipy.stats.stats as sss
import numpy
def create_all_possible_3vectors():
    return [[1,1,1],[0,1,1],[1,0,1],[0,0,1],[0,0,0],[1,1,0],[1,0,0],[0,1,0]]

def create_all_possible_vectors():
    return [[1,1,1,0],[0,1,1,0],[1,0,1,0],[0,0,1,0],[0,0,0,0],[1,1,0,0],[1,0,0,0],[0,1,0,0],[1,1,1,1],[0,1,1,1],[1,0,1,1],[0,0,1,1],[0,0,0,1],[1,1,0,1],[1,0,0,1],[0,1,0,1]]

def inner_product(w,x):
    return sum([a*b for a,b in zip(w,x)])

def get_rank(d1,d2,w):
    doc_scores = sorted([(d1,inner_product(w,d1)),(d2,inner_product(w,d2))],key=lambda x:(x[1],x[0]),reverse=True)
    ranked = [x[0] for x in doc_scores]
    return ranked[0],ranked[1],doc_scores[0][1],doc_scores[1][1]

def test():
    vectors = create_all_possible_3vectors()
    weights1 = [0.4,0.5,0.1]
    weights2 = [0.5,0.4,0.1]
    weights3 = [0.8,0.05,0.15]
    print determine_robustness_percent(weights1,vectors)
    print determine_robustness_percent(weights2,vectors)
    print determine_robustness_percent(weights3,vectors)




def get_ordered_indexes(w):
    indexes = []
    reference = copy(w)
    while w:
        maximum_weight_indexes = [i for i, x in enumerate(reference) if x == max(w)]
        max_val = max(w)
        indexes.extend(maximum_weight_indexes)
        w = [i for i in w if i != max_val]
    return indexes

def determine_average_changes_needed(w,vectors):
    total_changes = 0
    opposite = {}
    opposite[1] = 0
    opposite[0] = 1
    combinations = list(itertools.combinations(vectors, 2))
    denominator = len(combinations)
    indexes = get_ordered_indexes(copy(w))
    for combination in combinations:
        vector1 = combination[0]
        vector2=combination[1]
        winner,loser,winner_score,loser_score = get_rank(vector1,vector2,w)
        temp_loser = copy(loser)
        changes=0

        for index in indexes:
            if temp_loser[index]==-1:
                temp_loser[index]=1
                changes += 1
                new_score = inner_product(w,temp_loser)
                if new_score>=winner_score and new_score>loser_score:
                    break
        total_changes+=changes
    return float(total_changes)/denominator


def determine_robustness_percent(w,vectors):
    opposite={}
    opposite[1]=0
    opposite[0]=1
    file = open("res_min.tsv",'w')
    file.write('winner document\toriginal loser document\tnew document after change\twinner document score\tloser document score\tnew score after change\tchanged rank\n')
    numerator =0
    denominator =0
    combinations =  list(itertools.combinations(vectors, 2))
    for combination in combinations:
        vector1 = combination[0]
        vector2=combination[1]
        winner,loser,winner_score,loser_score = get_rank(vector1,vector2,w)
        able_to_change = False
        for index in range(4):
            loser_copy=copy(loser)
            loser_copy[index]=opposite[loser_copy[index]]
            current_score = inner_product(w,loser_copy)
            if current_score>=winner_score and current_score>loser_score:
                numerator += 1
                file.write(",".join([str(a) for a in winner])+"\t"+",".join([str(a) for a in loser])+"\t"+",".join([str(a) for a in loser_copy])+"\t"+str(winner_score)+"\t"+str(loser_score)+"\t"+str(current_score)+"\tYES\n")
                able_to_change=True
                break
        if not able_to_change:
            ""
            file.write(",".join([str(a) for a in winner])+"\t"+",".join([str(a) for a in loser])+"\t"+",".join([str(a) for a in loser])+"\t"+str(winner_score)+"\t"+str(loser_score)+"\t"+str(loser_score)+"\tNO\n")
        denominator+=1
    print numerator
    print denominator
    return float(numerator)/denominator

def pearsonr(x, y):
  # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(map(lambda x: pow(x, 2), x))
  sum_y_sq = sum(map(lambda x: pow(x, 2), y))
  psum = sum(imap(lambda x, y: x * y, x, y))
  num = psum - (sum_x * sum_y/n)
  den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if den == 0: return 0
  return num / den

def entropy_calculator(w):
    absoulute_weights_temp = [abs(a) for a in w]
    absoulute_weights = []
    for a in absoulute_weights_temp:
        if a==0:
            absoulute_weights.append(1.0/sys.maxint)
        else:
            absoulute_weights.append(a)
    summed_up = sum(absoulute_weights)
    p = [float(a)/summed_up for a in absoulute_weights]
    ent = 0
    base = len(p)

    for p_i in p:
        ent -= p_i*log(p_i,base)
    return ent


def check_terms(w):
    pairs = list(itertools.combinations(w, 2))
    for weight in w:
        for pair in pairs:
            if sum(pair)<=weight:
                return "NO"
    sums = [sum(pair) for pair in pairs]
    maximum = max(sums)
    index = sums.index(maximum)
    del sums[index]
    if maximum == max(sums):
        return "NO"
    return "YES"

if __name__=="__main__":
    vectors = create_all_possible_vectors()
    file = open("average_changes_four.tsv",'w')
    step=0.1
    sub_step=0.01
    file.write("WEIGHTS\tAVERAGE_CHANGES\tTERMS_SATISFIED\n")
    weights = [0.25,0.25,0.25,0.25]
    file.write(",".join([str(a) for a in weights]) + "\t" + str(determine_robustness_percent(weights, vectors))+"\t"+check_terms(copy(weights)) + "\n")
    weights = [0.3/1.3, 0.26/1.3, 0.24+(0.56/1.3)/1.3, 0.5/1.3]
    weights = [a/sum(weights) for a in weights]
    file.write(",".join([str(a) for a in weights]) + "\t" + str(determine_robustness_percent(weights, vectors)) + "\t" + check_terms(copy(weights)) + "\n")
    weights = [0.3, 0.26, 0.24, 0.2]
    file.write(",".join([str(a) for a in weights]) + "\t" + str(determine_robustness_percent(weights, vectors)) + "\t" + check_terms(copy(weights)) + "\n")
    """weights = [0.8,0.1,0.1,0]
    file.write(",".join([str(a) for a in weights]) + "\t" + str(
        determine_robustness_percent(weights, vectors)) + "\t" + check_terms(copy(weights)) + "\n")"""
    """weights = [0.8, 0.1, -0.1, 0]
    file.write(",".join([str(a) for a in weights]) + "\t" + str(
        determine_robustness_percent(weights, vectors)) + "\t" + check_terms(copy(weights)) + "\n")"""
    """weights=  [0.4,0.3,0.2,0.1]
    file.write(",".join([str(a) for a in weights]) + "\t" + str(
        determine_robustness_percent(weights, vectors)) + "\t" + check_terms(copy(weights)) + "\n")
    weights = [-0.2,0.26,-0.24,0.3]
    file.write(",".join([str(a) for a in weights]) + "\t" + str(
        determine_robustness_percent(weights, vectors)) + "\t" + check_terms(copy(weights)) + "\n")"""
    """while (weights[0]<=0):
        while(weights[1]>=0):
            file.write(",".join([str(a) for a in weights]) + "\t" + str(
            determine_average_changes_needed(weights, vectors)) + "\t" + check_terms(copy(weights)) + "\n")
            weights[1] -= sub_step
            weights[2] += sub_step
        weights[0]-=step
        weights[1]=1-weights[0]
        weights[2]=0"""
    file.close()



