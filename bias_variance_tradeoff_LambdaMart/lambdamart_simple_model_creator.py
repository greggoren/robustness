import preprocess_clueweb as p
from bias_variance_tradeoff_LambdaMart import single_model_handler_LambdaMart as mh
from bias_variance_tradeoff_LambdaMart import params
import sys

if __name__ == "__main__":
    preprocess = p.preprocess()
    trees = int(sys.argv[1])
    leaves = int(sys.argv[2])
    single_model_handler = mh.single_model_handler_LambdaMart(leaves, trees)
    qrels = params.qrels
    train_file = params.data_set_file
    single_model_handler.create_model_LambdaMart(trees, leaves, train_file, qrels)
    print("learning is finished")
