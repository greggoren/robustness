import preprocess_clueweb as p
from bias_variance_tradeoff_LambdaMart import single_model_handler_LambdaMart as mh
from bias_variance_tradeoff_LambdaMart import params
import sys
from functools import partial
from multiprocessing import Pool
if __name__ == "__main__":
    preprocess = p.preprocess()
    # trees = [(i + 1) * 10 for i in range(15, 46)]
    trees = 250
    leaves = [(1 + i) * 10 for i in range(5)]
    single_model_handler = mh.single_model_handler_LambdaMart(0, trees)
    qrels = params.qrels
    train_file = params.data_set_file
    f = partial(single_model_handler.create_model_LambdaMart, trees, train_file, qrels)
    with Pool(processes=5) as pool:
        # single_model_handler.create_model_LambdaMart(trees, )
        pool.map(f, leaves)
    leaves = [(1 + i) * 10 for i in range(5, 10)]
    single_model_handler = mh.single_model_handler_LambdaMart(0, trees)
    qrels = params.qrels
    train_file = params.data_set_file
    f = partial(single_model_handler.create_model_LambdaMart, trees, train_file, qrels)
    with Pool(processes=3) as pool:
        # single_model_handler.create_model_LambdaMart(trees, )
        pool.map(f, leaves)
    leaves = [(1 + i) * 10 for i in range(10, 13)]
    single_model_handler = mh.single_model_handler_LambdaMart(leaves, trees)
    qrels = params.qrels
    train_file = params.data_set_file
    f = partial(single_model_handler.create_model_LambdaMart, trees, train_file, qrels)
    with Pool(processes=2) as pool:
        # single_model_handler.create_model_LambdaMart(trees, )
        pool.map(f, leaves)
    leaves = [(1 + i) * 10 for i in range(13, 25)]
    single_model_handler = mh.single_model_handler_LambdaMart(leaves, trees)
    qrels = params.qrels
    train_file = params.data_set_file
    f = partial(single_model_handler.create_model_LambdaMart, trees, train_file, qrels)
    with Pool(processes=1) as pool:
        # single_model_handler.create_model_LambdaMart(trees, )
        pool.map(f, leaves)
    print("learning is finished")
