from scipy.stats import ttest_rel
import pickle

lb_rel_current = pickle.load(open("lb_query_rel_stats_current", 'rb'))
svm_rel_current = pickle.load(open("lb_query_rel_stats_current", 'rb'))
lb_metrics_current = pickle.load(open("lb_averaged_metrics_current", 'rb'))
svm_metrics_current = pickle.load(open("svm_averaged_metrics_current", 'rb'))

lb_rel_last = pickle.load(open("lb_query_rel_stats_last", 'rb'))
svm_rel_last = pickle.load(open("lb_query_rel_stats_last", 'rb'))
lb_metrics_last = pickle.load(open("lb_averaged_metrics_last", 'rb'))
svm_metrics_last = pickle.load(open("svm_averaged_metrics_last", 'rb'))

f = open("current_ttest", 'w')
for metric in lb_metrics_current:
    line = metric + " & $" + str(ttest_rel(lb_metrics_current[metric], svm_metrics_current[metric])[0]) + "$ & $" + str(
        ttest_rel(lb_metrics_current[metric], svm_metrics_current[metric])[1]) + "$\n"
    f.write(line)
for metric in lb_rel_current:
    line = metric + " & $" + str(ttest_rel(lb_rel_current[metric], svm_rel_current[metric])[0]) + "$ & $" + str(
        ttest_rel(lb_metrics_current[metric], svm_metrics_current[metric])[1]) + "$\n"
    f.write(line)
f.close()

f = open("last_ttest", 'w')
for metric in lb_metrics_last:
    line = metric + " & $" + str(ttest_rel(lb_metrics_last[metric], svm_metrics_last[metric])[0]) + "$ & $" + str(
        ttest_rel(lb_metrics_last[metric], svm_metrics_last[metric])[1]) + "$\n"
    f.write(line)
for metric in lb_rel_last:
    line = metric + " & $" + str(ttest_rel(lb_rel_last[metric], svm_rel_last[metric])[0]) + "$ & $" + str(
        ttest_rel(lb_metrics_last[metric], svm_metrics_last[metric])[1]) + "$\n"
    f.write(line)
f.close()
