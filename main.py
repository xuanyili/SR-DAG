from model.GaussianBIC_score import reward_GaussianBIC
from model.valid import ValidModel
from model.structure_learn import CombinatorialRL
from model.DataProcesser import BNSLDataset
from model.train import TrainModel
from model.BDeu_score import reward_BDeu
from model.BIC_score import reward_BIC
from experiments.utils import count_accuracy
import argparse
import logging
import torch
# import warnings
import random
import numpy as np
import pandas as pd
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='asia')
    parser.add_argument('--nodes_num', type=int, default='8')
    parser.add_argument('--size', type=int, default='4096')
    parser.add_argument('--score', type=str, default='BDeu')
    parser.add_argument('--loglevel', type=str, default='DEBUG')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--store', action='store_true')
    # asia 9178(2020):9187.0303
    parser.add_argument('--seed', type=int, default=9178)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--acc', action='store_true')
    parser.add_argument('--threshold', type=int, default=0.2)
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--global_score', action='store_true')
    parser.add_argument('--l1_loss', action='store_true')
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--acyc_w', type=int, default=500)
    parser.add_argument('--mask', action='store_true')

    opt = parser.parse_args()

    set_seed(opt.seed)
    dataset = opt.data
    seq_len = opt.nodes_num
    size = opt.size
    score = opt.score
    loglevel = opt.loglevel
    valid = opt.valid
    store = opt.store
    acc = opt.acc
    threshold = opt.threshold
    exp = opt.exp

    # os.environ['CUDA_VISIBLE_DEVICES']='1'

    formatter = logging.Formatter('%(asctime)s -- %(levelname)s: %(message)s')
    logger = logging.getLogger('SR-DAG_LOGGER')
    logger.setLevel(logging.DEBUG)

    filelog = logging.FileHandler('output/log/'+ dataset + '-' + score + str(valid) + '.log', 'w')
    if loglevel == 'DEBUG':
        filelevel = logging.DEBUG
    elif loglevel == 'INFO':
        filelevel = logging.INFO
    else:
        filelevel = logging.ERROR
    filelog.setLevel(filelevel)
    filelog.setFormatter(formatter)
    train_dataset = BNSLDataset(size, dataset=dataset, input_size=opt.input_size, use_file=True, exp=exp)
    logger.addHandler(filelog)

    if score == 'BDeu':
        reward = reward_BDeu(train_dataset.getdataset(), seq_len)
    elif score == 'BIC':
        reward = reward_BIC(train_dataset.getdataset(), seq_len)
    elif score == 'GaussBIC':
        reward = reward_GaussianBIC(train_dataset.getdataset(), seq_len)

    dot_model = CombinatorialRL(
            embedding_size = 1024,
            hidden_size = 1024,
            reward = reward,
            seq_len = seq_len,
            topk = opt.topk,
            mask = opt.mask,
            tanh_exploration = 10,
            use_tanh = True,
            input_size= opt.input_size,
            attention="Dot")
    if True:
        dot_model = dot_model.cuda()

    logger.info('start')
    if valid:
        model = torch.load('output/model/node'+ str(seq_len) + '_' + score + '.pth')
        # print(dot_model)
        dot_model.load_state_dict(model)
        dot_valid = ValidModel(dot_model, train_dataset, batch_size = opt.batch_size, threshold = threshold)
        G = dot_valid.valid_and_validate(1, logger, dataset + '-' + score + str(valid))

    else:
        dot_train = TrainModel(dot_model, train_dataset, batch_size = opt.batch_size, threshold = threshold, global_score=opt.global_score, l1_loss=opt.l1_loss, acyc_w=opt.acyc_w, mask = opt.mask)
        G = dot_train.train_and_validate(1, logger, dataset + '-' + score + str(valid), score, store)

    if acc:
        G_True = pd.read_csv('./data/real_graph/' + dataset + '.csv')
        acc = count_accuracy(G_True.to_numpy(), G.cpu().numpy().T)
        logger.info(acc)

    if exp:
        np.save("result/SR-DAG.npy", G.cpu().numpy().T)
        print(reward.compute_score(G).sum())
    logger.info('done')