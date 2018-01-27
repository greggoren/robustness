from scipy.stats import ttest_rel
import pickle

lb_rel_current = pickle.load(open("lb_query_rel_stats_current", 'rb'))
svm_rel_current = pickle.load(open("query_rel_stats", 'rb'))
lb_metrics_current = pickle.load(open("lb_averaged_metrics_current", 'rb'))
svm_metrics_current = pickle.load(open("svm_averaged_metrics", 'rb'))

# lb_rel_last = pickle.load(open("lb_query_rel_stats_last", 'rb'))
# svm_rel_last = pickle.load(open("query_rel_stats", 'rb'))
# lb_metrics_last = pickle.load(open("lb_averaged_metrics_last", 'rb'))
# svm_metrics_last = pickle.load(open("svm_averaged_metrics", 'rb'))
#
banned = ["032_0", "004_0", "032_1", "004_1", "002_1", "002_0"]
f = open("current_ttest", 'w')

for metric in lb_metrics_current:
    list1 = [lb_metrics_current[metric][q] for q in lb_metrics_current[metric] if q not in banned]
    list2 = [svm_metrics_current[metric][q] for q in svm_metrics_current[metric] if q not in banned]
    test = ttest_rel(list1, list2)
    line = metric + " & $" + str(round(test[0], 3)) + "$ & $" + str(round(test[1], 3)) + "$ \n"
    f.write(line)
for metric in lb_rel_current:
    list1 = [lb_rel_current[metric][q] for q in lb_rel_current[metric]]
    list2 = [svm_rel_current[metric][q] for q in svm_rel_current[metric]]
    test = ttest_rel(list1, list2)
    line = metric + " & $" + str(round(test[0], 3)) + "$ & $" + str(round(test[1], 3)) + "$ \n"
    f.write(line)
f.close()

# f = open("last_ttest", 'w')
# for metric in lb_metrics_last:
#     list1 = [lb_metrics_last[metric][q] for q in lb_metrics_last[metric]]
#     list2 = [svm_metrics_last[metric][q] for q in svm_metrics_last[metric]]
#     test = ttest_rel(list1, list2)
#     line = metric + " & $" + str(test[0]) + "$ & $" + str(test[1]) + "$\n"
#     f.write(line)
# for metric in lb_rel_last:
#     list1 = [lb_rel_last[metric][q] for q in lb_rel_last[metric]]
#     list2 = [svm_rel_last[metric][q] for q in svm_rel_last[metric]]
#     test = ttest_rel(list1, list2)
#     line = metric + " & $" + str(test[0]) + "$ & $" + str(
#         test[1]) + "$\n"
#     f.write(line)
# f.close()
