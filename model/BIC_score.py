import time
import numpy as np
import pandas as pd
from pgmpy.estimators import BaseEstimator
from pygobnilp.gobnilp import DiscreteBIC, DiscreteData
from torch.autograd import Variable
import torch
from scipy.special import gammaln
from math import lgamma, log


class reward_BIC():
    def __init__(self, sample_datas, node_size):
        self.data = pd.DataFrame(np.array(sample_datas))
        self.datatensor = sample_datas
        self.estimator = BaseEstimator(self.data)
        self.sample_size = len(self.data)
        self.node_size = node_size
        self.cache_score = {}
        self.node_unique = {}
        self.node_count = {}
        self.gobdata = DiscreteData(data_source=self.data.values.tolist(), varnames=self.data.columns, arities=None)
        self.gobscore = DiscreteBIC(self.gobdata)
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
    
    def compute_score(self, solution, USE_CUDA=False, USE_GOBNILP=False):
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
        if not parents:
            # node_data = np.unique(self.datatensor[:,[node]])
            node_data = self.node_unique[node]
            # print(node_data)
            num_parents_states = 1
            num_node_states = len(node_data)
            data = self.datatensor[:,[node]].numpy()
            counts = np.zeros((num_node_states, num_parents_states))
            for node_idx in range(num_node_states):
                # print((data == node_data[node_idx]))
                counts[node_idx,0] = self.node_count[node][node_idx]
            # print(counts)
        else:
            parents_data = np.unique(self.datatensor[:,parents], axis=0)
            parents_pos = {tuple(parents_data[i].tolist()):i for i in range(len(parents_data))}
            # print('parents_data',parents_data)
            # node_data = np.unique(self.datatensor[:,[node]])
            node_data = self.node_unique[node]
            node_pos = {node_data[i]:i for i in range(len(node_data))}
            # print('node_data',node_data)
            data = self.datatensor[:,[node] + parents].numpy()
            # print('data',data)
            num_parents_states = len(parents_data)
            num_node_states = len(node_data)
            counts = np.zeros((num_node_states, num_parents_states))
            unique_datas, unique_counts = np.unique(data, axis=0, return_counts=True)
            for uni_idx in range(len(unique_datas)):
                unique_data, unique_count = unique_datas[uni_idx], unique_counts[uni_idx]
                temp_node = unique_data[0]
                temp_pare = unique_data[1:]
                counts[node_pos[temp_node], parents_pos[tuple(temp_pare.tolist())]] = unique_count

            # for node_idx in range(num_node_states):
            #     for parents_idx in range(num_parents_states):
            #         # print('hstack',np.hstack((node_data[node_idx], parents_data[parents_idx])))
            #         counts[node_idx,parents_idx] = np.sum((data == np.hstack((node_data[node_idx], parents_data[parents_idx]))).all(1))
        # print(time.time() - start)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= 0.5 * log(self.sample_size) * num_parents_states * (num_node_states - 1)

        return score