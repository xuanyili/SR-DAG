from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import bnlearn as bn
import random
from sklearn import metrics
class BNSLDataset(Dataset):
    
    def __init__(self, num_samples, dataset = 'asia', input_size = 4000, use_file = True, exp = False):
        super(BNSLDataset, self).__init__()

        self.data_set = []
        self.input_size = input_size
        if exp:
            df = pd.read_csv("data/"+dataset+'/data.csv')
        elif use_file:
            df = pd.read_csv("./data/"+dataset+'.csv')
        else:
            self.model = bn.import_DAG(dataset)
            df = bn.sampling(self.model, n = num_samples)
        self.df = df
        self.data_set = torch.tensor(np.array(df))

        self.size = len(self.data_set)
        self.node_size = len(self.data_set[0])

        # for i in range(self.node_size):
        #     for j in range(self.node_size):
        #         if i==j:
        #             continue
        #         print(i,j,'mutual_info_score', metrics.mutual_info_score(self.data_set[:,i], self.data_set[:,j]))

    def __len__(self):
        return 500000

    def __getitem__(self, idx):
        ran = range(1, self.size)
        indexes = random.sample(ran, self.input_size)
        data = [self.data_set[i] for i in indexes]
        data = torch.stack(data, 0)
        #  return self.data_set[idx]
        return data
    
    def getdataset(self):
        return self.data_set
    
    def getnodesize(self):
        return self.node_size

    def getdf(self):
        return self.df