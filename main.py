import random as r
import params_ent
if __name__=="__main__":
    for gamma in params_ent.gammas:
        print(gamma)
        params_ent.score_file = str(gamma)+"_test.txt"
        