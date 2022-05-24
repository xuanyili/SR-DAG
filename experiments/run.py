"compare bnsl_rl, DAG_GNN, RL_BIC, RL_BIC2, GraN-DAG, notears, Gobnilp, PC"
import time
from pygobnilp.gobnilp import Gobnilp
from pgmpy.estimators import BDeuScore as bds
from pgmpy.estimators import PC
from utils import count_accuracy
from notears.linear import notears_linear
import pandas as pd
import numpy as np
import argparse
import logging
import sys
import torch
import os
import datetime
try:
    from model.BDeu_score import reward_BDeu
    from model.BIC_score import reward_BIC
    from model.GaussianBIC_score import reward_GaussianBIC
except:
    sys.path.append("../")
    from model.BDeu_score import reward_BDeu
    from model.BIC_score import reward_BIC
    from model.GaussianBIC_score import reward_GaussianBIC

def run_gobnilp(data, score='BDeu'):
    m = Gobnilp()

    m.learn(data_source=data.values.tolist(), varnames=data.columns, arities=None, score=score,  output_cpdag=False)
    bn = m.learned_bn
    nodes = data.columns
    w_g = np.zeros((len(data.columns), len(data.columns)))
    for (i,j) in bn.edges():
        w_g[list(nodes).index(i),list(nodes).index(j)] = 1
    return w_g

def run_pc(data):
    pc = PC(data)
    model = pc.estimate()
    nodes = data.columns
    w_g = np.zeros((len(data.columns), len(data.columns)))
    for (i,j) in model.edges():
        w_g[list(nodes).index(i),list(nodes).index(j)] = 1
    return w_g

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='synthetic_bn_n5_e10')
    parser.add_argument('--datadir', type=str, default='data/')
    parser.add_argument('--model', type=str, default='bnrl_bic')
    parser.add_argument('--datatype', type=str, default='discrete')

    opt = parser.parse_args()
    data = opt.data
    data_dir = opt.datadir
    model = opt.model
    datatype = opt.datatype

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    formatter = logging.Formatter('%(asctime)s -- %(levelname)s: %(message)s')
    logger = logging.getLogger('BNSL_RL_EXP_LOGGER')
    logger.setLevel(logging.DEBUG)

    filelog = logging.FileHandler('result/log/'+ data + '.log')
    filelevel = logging.DEBUG
    filelog.setLevel(filelevel)
    filelog.setFormatter(formatter)
    logger.addHandler(filelog)

    datadf = pd.read_csv(data_dir+data+"/data.csv")
    print()
    true_g = np.load(data_dir+data+"/DAG.npy")
    datanp = np.load(data_dir+data+"/data.npy")

    if datatype == 'discrete':
        reward_bdeu = reward_BDeu(torch.tensor(datanp), len(datadf.columns))
        reward_bic = reward_BIC(torch.tensor(datanp), len(datadf.columns))
    elif datatype == 'continuous':
        reward_gaussbic = reward_GaussianBIC(torch.tensor(datanp), len(datadf.columns))

    #PC
    if model == 'pc':
        start = time.time()
        g_pc = run_pc(datadf)
        end = time.time()
        exec_time = (end - start)
        acc_pc = count_accuracy(true_g, g_pc)
        scorebdeu = reward_bdeu.compute_score(torch.tensor(g_pc.T))
        scorebic = reward_bic.compute_score(torch.tensor(g_pc.T))
        logger.info("PC          acc:{}, exec time:{}, BDeu score:{}, BIC score:{}".format(acc_pc, exec_time, scorebdeu.sum(), scorebic.sum()))

    #gobnilp BDeu
    if model == 'gobnilp_bdeu':
        start = time.time()
        g_gobnilp_bdeu = run_gobnilp(datadf, 'BDeu')
        end = time.time()
        exec_time = (end - start)
        acc_gobnilp_bdeu = count_accuracy(true_g, g_gobnilp_bdeu)
        score = reward_bdeu.compute_score(torch.tensor(g_gobnilp_bdeu.T))
        logger.info("gobnilpBDeu acc:{}, exec time:{}, BDeu score:{}".format(acc_gobnilp_bdeu, exec_time, score.sum()))

    #gobnilp BIC
    if model == 'gobnilp_bic':
        start = time.time()
        g_gobnilp_bic = run_gobnilp(datadf, 'DiscreteBIC')
        end = time.time()
        exec_time = (end - start)
        acc_gobnilp_bic = count_accuracy(true_g, g_gobnilp_bic)
        score = reward_bic.compute_score(torch.tensor(g_gobnilp_bic.T))
    
        logger.info("gobnilpBIC  acc:{}, exec time:{}, BIC score:{}".format(acc_gobnilp_bic, exec_time, score.sum()))

    #gobnilp GuassBIC
    if model == 'gobnilp_gaussbic':
        start = time.time()
        g_gobnilp_bic = run_gobnilp(datadf, 'GaussianBIC')
        end = time.time()
        exec_time = (end - start)
        acc_gobnilp_bic = count_accuracy(true_g, g_gobnilp_bic)
        score = reward_gaussbic.compute_score(torch.tensor(g_gobnilp_bic.T))
    
        logger.info("gobnilpBIC  acc:{}, exec time:{}, BIC score:{}".format(acc_gobnilp_bic, exec_time, score.sum()))

    #SR-DAG BDeu
    if model == 'sr_bdeu':
        start = time.time()
        os.system("python ../main.py --data {} --nodes_num {} --exp --store --batch_size 10".format(data, len(datadf.columns)))
        end = time.time()
        exec_time = (end - start)
        g_bnrl_bdeu = np.load("result/SR-DAG.npy", allow_pickle = True)
        acc_bnrl_bdeu = count_accuracy(true_g, g_bnrl_bdeu)
        score = reward_bdeu.compute_score(torch.tensor(g_bnrl_bdeu.T))
        logger.info("SRDAG BDeu  acc:{}, exec time:{}, BDeu score:{}".format(acc_bnrl_bdeu, exec_time, score.sum()))
    
    #SR-DAG BDeu
    if model == 'sr_bdeu_mask':
        start = time.time()
        os.system("python ../main.py --data {} --nodes_num {} --exp --store --batch_size 10 --mask".format(data, len(datadf.columns)))
        end = time.time()
        exec_time = (end - start)
        g_bnrl_bdeu = np.load("result/SR-DAG.npy", allow_pickle = True)
        acc_bnrl_bdeu = count_accuracy(true_g, g_bnrl_bdeu)
        score = reward_bdeu.compute_score(torch.tensor(g_bnrl_bdeu.T))
        logger.info("SRDAG BDeu  acc:{}, exec time:{}, BDeu score:{}".format(acc_bnrl_bdeu, exec_time, score.sum()))

    #SR-DAG BIC
    if model == 'sr_bic_mask':
        start = time.time()
        os.system("python ../main.py --data {} --nodes_num {} --score BIC --exp --batch_size 10 --mask".format(data, len(datadf.columns)))
        end = time.time()
        exec_time = (end - start)
        g_bnrl_bic = np.load("result/SR-DAG.npy", allow_pickle = True)
        acc_bnrl_bic = count_accuracy(true_g, g_bnrl_bic)
        score = reward_bic.compute_score(torch.tensor(g_bnrl_bic.T))
        logger.info("SRDAG BIC M acc:{}, exec time:{}, BIC score:{}".format(acc_bnrl_bic, exec_time, score.sum()))

    #SR-DAG BIC
    if model == 'sr_bic':
        start = time.time()
        os.system("python ../main.py --data {} --nodes_num {} --score BIC --exp --batch_size 1 --acyc_w 10".format(data, len(datadf.columns)))
        end = time.time()
        exec_time = (end - start)
        g_bnrl_bic = np.load("result/SR-DAG.npy", allow_pickle = True)
        acc_bnrl_bic = count_accuracy(true_g, g_bnrl_bic)
        score = reward_bic.compute_score(torch.tensor(g_bnrl_bic.T))
        logger.info("SRDAG BIC   acc:{}, exec time:{}, BIC score:{}".format(acc_bnrl_bic, exec_time, score.sum()))

    #notear
    if model == 'notear':
        start = time.time()
        W_notears = notears_linear(datanp,lambda1=0.1, loss_type='logistic')
        end = time.time()
        exec_time = (end - start)
        g_notears = (W_notears != 0)
        acc_notear = count_accuracy(true_g, g_notears)
        if datatype == 'continuous':
            scorebic = reward_gaussbic.compute_score(torch.tensor(g_notears.T))
            logger.info("No Tears    acc:{}, exec time:{}, BIC score:{}".format(acc_notear, exec_time, scorebic.sum()))
        elif datatype == 'discrete':
            scorebdeu = reward_bdeu.compute_score(torch.tensor(g_notears.T))
            scorebic = reward_bic.compute_score(torch.tensor(g_notears.T))
            logger.info("No Tears    acc:{}, exec time:{}, BDeu score:{}, BIC score:{}".format(acc_notear, exec_time, scorebdeu.sum(), scorebic.sum()))

    #RL_BIC
    if model == 'rlbic':
        start = time.time()
        os.system("python Causal_Discovery_RL/src/main.py  --max_length {}  --data_size 4096 --score_type BIC_different_var \
        --reg_type LR --read_data --transpose --data_path data/{}  \
            --lambda_flag_default --nb_epoch 20000 --input_dimension 64 --lambda_iter_num 1000\
                --output_path rlbic/{}".format(len(datadf.columns), data, data))
        end = time.time()
        exec_time = (end - start)
        g_rl_bic = np.load("output/rlbic/{}/graph/pred_G.npy".format(data), allow_pickle = True)
        acc_rl_bic = count_accuracy(true_g, g_rl_bic)
        scorebdeu = reward_bdeu.compute_score(torch.tensor(g_rl_bic.T))
        scorebic = reward_bic.compute_score(torch.tensor(g_rl_bic.T))
        logger.info("RL BIC      acc:{}, exec time:{}, BDeu score:{}, BIC score:{}".format(acc_rl_bic, exec_time, scorebdeu.sum(), scorebic.sum()))

    #RL_BIC2
    if model == 'rlbic2':
        start = time.time()
        os.system("python Causal_Discovery_RL/src/main.py  --max_length {}  --data_size 4096 --score_type BIC \
        --reg_type LR --read_data --transpose --data_path data/{}  \
            --lambda_flag_default --nb_epoch 20000 --input_dimension 64 --lambda_iter_num 1000\
                --output_path rlbic2/{}".format(len(datadf.columns), data, data))
        end = time.time()
        exec_time = (end - start)
        g_rl_bic2 = np.load("output/rlbic2/{}/graph/pred_G.npy".format(data), allow_pickle = True)
        acc_rl_bic2 = count_accuracy(true_g, g_rl_bic2)
        scorebdeu = reward_bdeu.compute_score(torch.tensor(g_rl_bic2.T))
        scorebic = reward_bic.compute_score(torch.tensor(g_rl_bic2.T))
        logger.info("RL BIC2     acc:{}, exec time:{}, BDeu score:{}, BIC score:{}".format(acc_rl_bic2, exec_time, scorebdeu.sum(), scorebic.sum()))

    # #DAG_GNN
    if model == 'dag_gnn':
        start = time.time()
        os.system("python DAG-GNN/src/train.py --data_dir data/{} --data_variable_size {}".format(data, len(datadf.columns)))
        end = time.time()
        exec_time = (end - start)
        g_daggnn = np.loadtxt("predG")
        g_daggnn[np.abs(g_daggnn) < 0.1] = 0
        g_daggnn = (g_daggnn !=0)
        acc_daggnn = count_accuracy(true_g, g_daggnn)
        scorebdeu = reward_bdeu.compute_score(torch.tensor(g_daggnn.T))
        scorebic = reward_bic.compute_score(torch.tensor(g_daggnn.T))
        logger.info("DAG-GNN     acc:{}, exec time:{}, BDeu score:{}, BIC score:{}".format(acc_daggnn, exec_time, scorebdeu.sum(), scorebic.sum()))

    #GraN-DAG
    if model == 'gran_dag':
        start = time.time()
        os.system("python GraN-DAG/main.py  --data-path data/{} --model NonLinGaussANM --train --to-dag --num-vars {} --exp-path exp/{} --jac-thresh".format(data, len(datadf.columns), data))
        end = time.time()
        exec_time = (end - start)
        g_gran = np.load("exp/{}/to-dag/DAG.npy".format(data), allow_pickle = True)
        acc_gran = count_accuracy(true_g, g_gran)
        scorebdeu = reward_bdeu.compute_score(torch.tensor(g_gran.T))
        scorebic = reward_bic.compute_score(torch.tensor(g_gran.T))
        logger.info("GraN-DAG    acc:{}, exec time:{}, BDeu score:{}, BIC score:{}".format(acc_gran, exec_time, scorebdeu.sum(), scorebic.sum()))

