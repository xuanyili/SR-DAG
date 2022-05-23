from pickle import TRUE
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import random
import time
import torch.nn.functional as F
import  torch.multiprocessing as mp

def is_nan(param_model):
    is_nan = False

    if torch.isnan(param_model.weight).sum() > 0:
        is_nan = True
        # break
    return is_nan

class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=TRUE):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size)) 
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  
        embedded = []
        inputs = inputs.unsqueeze(1) 
        for i in range(seq_len):
            embedded.append(torch.tanh(torch.bmm(inputs[:, :, :, i].float(), embedding)))
        embedded = torch.cat(embedded, 1)
        return embedded

class NewGraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, seq_len, use_cuda=TRUE):
        super(NewGraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Linear(input_size, embedding_size)
        self.nodeembedding = nn.Linear(seq_len, seq_len)
        # self.embedding.weight.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        # embedding = self.embedding.repeat(batch_size, 1, 1)  
        # embedded = []
        # inputs = inputs.unsqueeze(1)
        # for i in range(seq_len):
        #     embedded.append(torch.tanh(torch.bmm(inputs[:, :, :, i].float(), embedding)))
        # embedded = torch.cat(embedded, 1)
        embedded = self.nodeembedding(inputs.float())
        embedded = self.embedding(embedded.transpose(1,2))

        return embedded

class PointerNet(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            topk,
            mask,
            tanh_exploration,
            use_tanh,
            attention,
            use_cuda=TRUE,
            input_size = 4000):
        super(PointerNet, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.seq_len        = seq_len
        self.use_cuda       = use_cuda
        self.topk = topk
        self.mask = mask
        
        self.embedding = GraphEmbedding(input_size, embedding_size, use_cuda=use_cuda)
        # self.embedding = NewGraphEmbedding(input_size, embedding_size, seq_len, use_cuda=use_cuda)
        self.linearlist = nn.ModuleList(nn.Linear(hidden_size, seq_len) for i in range(seq_len))
        self.linear = nn.Linear(hidden_size, seq_len)
        self.node = nn.Embedding(seq_len, hidden_size)
        
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        self.x = [i for i in range(seq_len)]
        random.shuffle(self.x)

    def mask_acyalic_rec(self, mask, idx, batch_id, prev_sols):
            childs = prev_sols[batch_id, :, idx].nonzero()
            # print(childs)
            mask[batch_id, childs] = 1
            for chidx in childs.squeeze(1):

                # print("batch_id:{} idx:{} chidx:{}".format(batch_id, idx, childs))
                self.mask_acyalic_rec(mask, chidx, batch_id, prev_sols)

    def masks_acyalic(self, mask, idx, prev_sols):
        batch_size = prev_sols.size(0)
        clone_mask = mask.clone()
        # pas = prev_sols[[i for i in range(batch_size)], idx, :]
        for i in range(batch_size):
            self.mask_acyalic_rec(clone_mask, idx, i, prev_sols)
        return clone_mask
        
    # def compute_mask(self, mask, idx, prev_sols):
    #     clone_mask = mask.clone()
    #     batch_size = prev_sols.size(0)
    #     for i in range(batch_size):
    #         parents = prev_sols[i,idx,:].nonzero().squeeze(1)
    #         # print(parents)
    #         for parent in parents:
    #             clone_mask[i, parent, idx] = 1
    #             masktmp = clone_mask[i,idx,:].nonzero().squeeze(1)
    #             if len(masktmp) > 0:
    #                 clone_mask[i, parent, masktmp] = 1     
    #     return clone_mask

    def compute_mask(self, mask, idx, prev_sols):
        clone_mask = mask.clone()
        batch_size = prev_sols.size(0)
        for i in range(batch_size):
            parents = prev_sols[i,idx,:].nonzero().squeeze(1)
            # print(parents)
            for parent in parents:
                clone_mask[i, parent, idx] = 1
                masktmp = clone_mask[i,:,parent].nonzero().squeeze(1)
                if len(masktmp) > 0:
                    clone_mask[i, masktmp, idx] = 1
                masktmp2 = clone_mask[i,idx,:].nonzero().squeeze(1) 
                if len(masktmp2) > 0 :
                    pa_tmp = clone_mask[i,:,idx].nonzero().squeeze(1)
                    for pa in pa_tmp:
                        clone_mask[i, pa, masktmp2] = 1
        return clone_mask

    def apply_mask_to_logits(self, logits_1,  mask, idx, prev_sols): 
        clone_mask = mask.clone()
        # if idx > 0 :
        #     self.compute_mask(clone_mask, idx-1, prev_sols)
        logits_1[clone_mask[:,idx,:]] = -np.inf
        return logits_1, clone_mask

    def apply_mask_to_actions(self, action,  mask, idx): 
        clone_mask = mask.clone()
        # if idx > 0 :
        #     self.compute_mask(clone_mask, idx-1, prev_sols)
        action[clone_mask[:,idx,:]] = 0
        return action, clone_mask

    def forward(self, inputs, offset_prob):
        """
        Args: 
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        assert seq_len == self.seq_len

        prev_probs = torch.zeros(batch_size, seq_len, seq_len)
        prev_probs_2 = torch.zeros(batch_size, seq_len, seq_len)
        prev_idxs = torch.zeros(batch_size, seq_len, seq_len).bool()
        mask = torch.zeros(batch_size, seq_len, seq_len).bool()
        node_mask = torch.zeros(seq_len).bool()
        if self.use_cuda:
            mask = mask.cuda()
            node_mask = node_mask.cuda()
            prev_probs = prev_probs.cuda()
            prev_probs_2 = prev_probs_2.cuda()
            prev_idxs = prev_idxs.cuda()
       
        embedded = self.embedding(inputs)

        x = [i for i in range(seq_len)]
        # if self.mask:
        #     random.shuffle(x)
        for idx in x:
            # logits = self.linearlist[idx](self.node(torch.LongTensor((1,)).fill_(idx).cuda()))

            node = embedded[[i for i in range(batch_size)], idx, :]
            logits = self.linearlist[idx](node)

            cpu_node = self.node(torch.LongTensor((1,)).fill_(idx).cuda()).detach().cpu().numpy()
            cpu_logits = logits.detach().cpu().numpy()
            # print(logits.shape)
            # print(mask.shape)
            # logits, mask = self.apply_mask_to_logits(logits, mask, idx, prev_idxs)

            # logits[:,:,1], mask[:, idx, :] = self.apply_mask_to_logits(logits[:,:,1], mask[:, idx, :], idx, prev_idxs)
            probs = torch.clamp(F.sigmoid(logits).unsqueeze(2), min=1e-4, max=1-1e-4)  #batch_size*seq_len
            cpu_probs = probs.detach().cpu().numpy()
            # probs = torch.clamp(F.relu(probs - offset_prob), min=1e-4, max=1-1e-4)
            _, sorted_probs_idx = probs.sort(descending=True, dim=-2)
            idxs = probs[:,:,0].bernoulli()

            if self.topk != 0:
                for i in range(batch_size):
                    for j in range(self.topk, seq_len):
                        idxs[i,sorted_probs_idx[i][j]] = 0
            if self.mask:
                idxs, mask = self.apply_mask_to_actions(idxs, mask, idx)
            # for i in range(len(probs)):
            #     for j in range(len(probs[i])):
            #         if idxs[i,j] == 1 and probs[i,j,0] < 0.1:
            #             idxs[i,j] = 0
            #         elif idxs[i,j] == 0 and probs[i,j,0] > 0.9:
            #             idxs[i,j] = 1                    
            idxs[:,idx] = 0
            try:
                prev_probs[[i for i in range(batch_size)], idx, :] = probs[:,:,0] #batch_size*seq_len*seq_len
            except:
                print(cpu_node)
                print(cpu_logits)
                print(cpu_probs)
                prev_probs[[i for i in range(batch_size)], idx, :] = probs[:,:,0]
            prev_idxs[[i for i in range(batch_size)], idx, :] = idxs.bool()
            if self.mask:
                mask = self.compute_mask(mask, idx, prev_idxs)
        return prev_probs, prev_probs_2, prev_idxs, mask

class CombinatorialRL(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            reward,
            seq_len,
            topk,
            mask,
            tanh_exploration,
            use_tanh,
            input_size,
            attention,
            use_cuda=TRUE):
        super(CombinatorialRL, self).__init__()
        self.reward = reward
        self.use_cuda = use_cuda
        
        self.actor = PointerNet(
                embedding_size,
                hidden_size,
                seq_len,
                topk,
                mask,
                tanh_exploration,
                use_tanh,
                attention,
                use_cuda,
                input_size)


    def forward(self, inputs, offset_prob=0):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        # input_size = inputs.size(1)
        seq_len    = inputs.size(2)
        
        probs, probs_2, action_idxs, mask = self.actor(inputs, offset_prob)
        # self.reward = reward_BDeu(inputs.cpu(), seq_len)
        # print("actor done")
        # actions = []
        # inputs = inputs.transpose(1, 2)
        # # print("inputs:{}".format(inputs))
        # for action_id in action_idxs:
        #     actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])

            
        # action_probs = []    
        # for prob, action_id in zip(probs, action_idxs):
        #     action_probs.append(prob[[x for x in range(batch_size)], action_id.data])
        
        # probs = probs.masked_select(action_idxs)
        # print("compute reward")
        # start = time.time()
        R = self.reward.reward(action_idxs)
        # end = time.time()
        # print(end - start)
        # print(action_idxs)
        # print(R)
        return R, probs, probs_2, action_idxs, mask