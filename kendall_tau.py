import itertools


def determine_order(pair, ranked_list):
    tmp = list(pair)
    return sorted(tmp, key=lambda x: ranked_list.index(x))


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
    if metric == "max":
        return max(w1, w2)
    if metric == "mean":
        return float(w1 + w2) / 2
        # if metric == "weighted":
        #     return (float(1)/4)*w1+(float(3)/4)*w2


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
