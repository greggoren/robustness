import analyze_competition as ac
import pickle
if __name__=="__main__":
    f = open("data/scores.pickle",'rb')
    s_nd, s_map, e_nd, e_map = pickle.load(f)
    ac.create_plot("MAP values","plt/map.jpg","EPOCH","MAP",s_map,e_map,range(1,9))
    ac.create_plot("NDCG@5 values","plt/nd.jpg","EPOCH","NDCG@5",s_nd,e_nd,range(1,9))