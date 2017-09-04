import analyze_competition as ac
import pickle
if __name__=="__main__":
    a = ac.analysis()
    s_nd,s_map=a.calculate_metrics("SVM")
    e_nd, e_map = a.calculate_metrics("SVM_ENT")
    with open("scores.pickle",'wb') as f:
        pickle.dump((s_nd,s_map,e_nd,e_map),f)
