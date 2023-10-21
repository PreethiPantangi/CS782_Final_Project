import argparse

from recommendation.algorithms.Ceaser.interactions import Interactions
from recommendation.algorithms.Ceaser.utils import *
from recommendation.algorithms.Ceaser.train_caser import Recommender

def callCeaser():
    print("In __main__")
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='./recommendation/datasets/movielens/test/train.txt')
    parser.add_argument('--test_root', type=str, default='./recommendation/datasets/movielens/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    model_config = model_parser.parse_args()
    model_config.L = config.L

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(config)
    print(model_config)
    # fit model
    model = Recommender(n_iter=config.n_iter, batch_size=config.batch_size, learning_rate=config.learning_rate, l2=config.l2, neg_samples=config.neg_samples, model_args=model_config, use_cuda=config.use_cuda)

    model.fit(train, test, verbose=True)