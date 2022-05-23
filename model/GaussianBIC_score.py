import time
import numpy as np
import pandas as pd
from pgmpy.estimators import BaseEstimator
from pygobnilp.gobnilp import GaussianBIC, ContinuousData
from torch.autograd import Variable
import torch
from scipy.special import gammaln
from math import lgamma, log


class reward_GaussianBIC():
    def __init__(self, sample_datas, node_size):
        self.data = pd.DataFrame(np.array(sample_datas))
        self.datatensor = sample_datas
        self.estimator = BaseEstimator(self.data)
        self.sample_size = len(self.data)
        self.node_size = node_size
        self.cache_score = {}
        self.node_unique = {}
        self.node_count = {}
        print(sample_datas.shape)
        print(node_size)
        self.gobdata = ContinuousData(self.data.values.tolist(), varnames=self.data.columns)
        self.gobscore = GaussianBIC(self.gobdata)
        for node in range(self.node_size):
            node_data, node_count = np.unique(self.datatensor[:,[node]], return_counts=True)
            self.node_unique[node] = node_data
            self.node_count[node] = node_count
    
    def reward(self, sample_solution, USE_CUDA=False):
        """
        Args:
            sample_solution node_size*node_size of [batch_size]
        """
        batch_size = sample_solution.size(0)
        scores = torch.zeros([batch_size, self.node_size])
        if USE_CUDA:
            scores = scores.cuda()

        for i in range(batch_size):
            # print("before compute_score")
            scores[i,:] = self.compute_score(sample_solution[i,:,:], USE_CUDA)
            # print("after compute_score")
        
        return scores
    
    def compute_score(self, solution, USE_CUDA=False, USE_GOBNILP=True):
        # print(solution.shape)
        assert self.node_size == solution.size(0)
        assert self.node_size == solution.size(1)
        score =  torch.zeros(self.node_size)
        # range_list = torch.arange(self.node_size).unsqueeze(1)
        # if USE_CUDA:
        #     range_list = range_list.cuda()
        # for node in range_list:
        #     print(node)
        for node in range(self.node_size):
            parents = torch.nonzero(solution[node]).squeeze(1).cpu().numpy().tolist()
            if USE_GOBNILP:
                score[node], _ = self.gobscore.score(node, tuple(parents))
            else:
                cache_key = [node] + parents
                cache_key = tuple(cache_key)
                if cache_key not in self.cache_score:
                    score[node] = self.local_score_numpy(node, parents)
                    self.cache_score[cache_key] = score[node]
                else:
                    score[node] = self.cache_score[cache_key]
                # parents = torch.nonzero(solution[node]).squeeze(1)
                # new_node = torch.tensor([node])
                # if USE_CUDA:
                #     new_node = new_node.cuda()
                # score[node] = self.local_score_torch(new_node, parents, USE_CUDA)
            # print("compute score done")
        return score

    def local_score_numpy(self, node, parents):
        #developing
        return 0