import itertools
import numpy as np

def determine_order(pair, ranked_list):
    tmp = list(pair)
    return sorted(tmp, key=lambda x: ranked_list.index(x))


def kendall_distance(ranked1, ranked2):
    discordant = 0
    all_pairs = list(itertools.combinations(ranked1, 2))
    for pair in all_pairs:
        winner1, loser1 = determine_order(pair, ranked1)
        winner2, loser2 = determine_order(pair, ranked2)
        if winner1 != winner2:
            discordant += 1
    return float(discordant) / len(all_pairs)


def weighted_kendall_distance(ranked1, ranked2, weights, metric):
    discordant = 0
    all_pairs = list(itertools.combinations(ranked1, 2))
    for pair in all_pairs:
        winner1, loser1 = determine_order(pair, ranked1)
        winner2, loser2 = determine_order(pair, ranked2)
        if winner1 != winner2:
            discordant += float(1) / (metric_enforcer(metric, weights[loser1], weights[winner1]) + 1)
    return float(discordant)


def normalized_weighted_kendall_distance(ranked1, ranked2, weights, cd, metric):
    discordant = 0
    all_pairs = list(itertools.combinations(ranked1, 2))
    for pair in all_pairs:
        winner1, loser1 = determine_order(pair, ranked1)
        winner2, loser2 = determine_order(pair, ranked2)
        if winner1 != winner2:
            discordant += float(1) / (
                normalzaied_metric_enforcer(metric, weights[loser1], weights[winner1], cd[loser2], cd[winner2]) + 1)
    return float(discordant)


def kendall_tau(ranked1, ranked2):
    concordant = 0
    discordant = 0
    all_pairs = list(itertools.combinations(ranked1, 2))
    for pair in all_pairs:
        winner1, loser1 = determine_order(pair, ranked1)
        winner2, loser2 = determine_order(pair, ranked2)
        if winner1 != winner2:
            discordant += 1
        else:
            concordant += 1
    return float(concordant - discordant) / len(all_pairs)


def metric_enforcer(metric, w1, w2):
    if metric == "diff":
        v1 = np.linalg.norm(w1)
        v2 = np.linalg.norm(w2)
        return abs(v2 - v1)
    if metric == "rel":
        return np.linalg.norm(w2 - w1)
    if metric == "sum":
        v1 = np.linalg.norm(w1)
        v2 = np.linalg.norm(w2)
        return v1 + v2


def normalzaied_metric_enforcer(metric, w1, w2, d1, d2):
    if metric == "diff":
        v1 = np.linalg.norm(w1) / np.linalg.norm(d1)
        v2 = np.linalg.norm(w2) / np.linalg.norm(d2)
        return abs(v2 - v1)
    if metric == "rel":
        if np.linalg.norm(d2 - d1) == 0:
            return ""
        return np.linalg.norm(w2 - w1) / (np.linalg.norm(d2 - d1))
    if metric == "sum":
        v1 = np.linalg.norm(w1) / np.linalg.norm(d1)
        v2 = np.linalg.norm(w2) / np.linalg.norm(d2)
        return v1 + v2


def weighted_kendall_tau(ranked1, ranked2, weights, metric):
    concordant = 0
    discordant = 0
    denominator = 0
    all_pairs = list(itertools.combinations(ranked1, 2))
    for pair in all_pairs:
        winner1, loser1 = determine_order(pair, ranked1)
        winner2, loser2 = determine_order(pair, ranked2)
        if winner1 != winner2:
            discordant += float(1) / (metric_enforcer(metric, weights[loser1], weights[winner1]) + 1)
        else:
            concordant += float(1) / (metric_enforcer(metric, weights[loser1], weights[winner1]) + 1)
        denominator += float(1) / (metric_enforcer(metric, weights[loser1], weights[winner1]) + 1)
    return float(concordant - discordant) / denominator
