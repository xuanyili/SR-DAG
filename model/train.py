from inspect import trace
import torch
import torch.optim as optim
import torch.autograd as autograd
import time
try:
    from DataProcesser import DataLoader
except:
    from .DataProcesser import DataLoader
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import scipy.linalg as slin


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.cpu().detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E).to(input.device)
        ctx.save_for_backward(E)
        return torch.tensor(f, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input

trace_expm = TraceExpm.apply

def is_non_acyclic(input):
    E = slin.expm(input.cpu().detach().numpy())
    return np.trace(E) - input.shape[1]

def is_nan(named_params_model):
    is_nan = False
    for (name_model, param_model) in named_params_model:
        if param_model.grad is not None:
            if torch.isnan(param_model.grad).sum() > 0:
                print('RRRRRRRRRRRRRRRRRRRR')
                is_nan = True
                break
        if torch.isnan(param_model.data).sum() > 0:
            print('PPPPPPPPPPPPPPPPP')
            is_nan = True
            break
    return is_nan

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return 0.5 + global_step / warmup_step * 2
    else:
        return 1.0

class TrainModel:
    def __init__(self, model, train_dataset, batch_size=10, threshold=0.2, max_grad_norm=2., global_score=False, l1_loss=True, acyc_w=500, mask=False, USE_CUDA = True):
        self.model = model
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.seq_len = train_dataset.getnodesize()
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        self.actor_optim   = optim.Adam(model.actor.parameters(), lr=1e-4)
        # self.actor_optim   = optim.SGD(model.actor.parameters(), lr=1e-2)
        self.max_grad_norm = max_grad_norm
        
        self.train_tour = []
        self.val_tour   = []
        self.losses = []
        self.USE_CUDA = USE_CUDA
        self.epochs = 0
        self.prob_warm = 0
        self.global_score = global_score
        self.l1_loss_func = torch.nn.L1Loss(reduction='sum')
        self.l1_loss = l1_loss
        self.total_actions = torch.zeros([self.seq_len, self.seq_len]).cuda()
        self.acyclic_w = acyc_w
        self.mask = mask
        # torch.autograd.set_detect_anomaly(True)
    def dfs(self, v):
        if v in self.vis:
            if v in self.trace:
                v_index = self.trace.index(v)
                print("True:")
                for i in range(v_index, len(trace)):
                    print(trace[i] + ' ', end='')
                print(v)
                print('\n')
                return True
            return True

        self.vis.append(v)
        self.trace.append(v)
        for vs in self.a[v]:
            self.dfs(vs)
        self.trace.pop()
        return False
   
    def train_and_validate(self, n_epochs, logger, filename='none', score='BDeu', store=False):
        max_reward = torch.zeros(1)
        mean_reward = torch.zeros(1)
        rewards = []
        losses =[]
        first = True
        max_tmp = torch.zeros(1)
        count = 0

        if self.USE_CUDA: 
            max_reward = max_reward.cuda()
            mean_reward = mean_reward.cuda()
            max_tmp = max_tmp.cuda()
        self.model.train()
        for epoch in range(n_epochs):
            for batch_id, sample_batch in enumerate(self.train_loader):
                start = time.time()
                self.actor_optim.zero_grad()
                inputs = sample_batch
                if self.USE_CUDA:
                    inputs = inputs.cuda()
                # print(inputs)
                offset_prob = (1 - warmup_linear(batch_id, self.prob_warm))
                R, probs, probs_2, actions_idxs, mask = self.model(inputs, offset_prob)
                # mean_reward = R.mean(0)
                rewards.append(R)
                mean_reward = torch.cat(rewards).mean(0)
                sum_reward = R.sum(1)
                _, sorted_reward_idx = sum_reward.sort(descending=True)
                for idx in sorted_reward_idx:
                    if is_non_acyclic(actions_idxs[idx]) < 1e-10:
                        # if self.acyclic_w > 100:
                        #     self.acyclic_w -= 1
                        max_reward_idx = idx
                        if first:
                            first = False

                            max_reward = R[max_reward_idx]
                            max_actions = actions_idxs[max_reward_idx]
                        else:
                            if sum_reward[max_reward_idx] >= max_reward.sum():
                                max_reward = R[max_reward_idx]
                                max_actions = actions_idxs[max_reward_idx]
                        break
                    # else:
                        # self.acyclic_w += 1
                self.total_actions += actions_idxs.sum(0)
                # else:
                #     self.vis = []
                #     self.trace = []
                #     self.a = {}
                #     for i in range(self.seq_len):
                #         temp = actions_idxs[max_reward_idx][i].nonzero().squeeze(1).cpu().detach().tolist()
                #         self.a[i] = temp
                #         # print(self.a)
                #     print(self.dfs(0))
                # else:
                #     critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean(0))

                if len(max_reward) == 1:
                    advantage = (R - mean_reward)
                else:
                    advantage = (R - max_reward)
                # advantage = (first_reward - R)

                logprobs = torch.zeros([self.batch_size, self.seq_len])
                logprobs2 = torch.zeros([self.batch_size, self.seq_len])
                acyclic_loss = torch.zeros([self.batch_size])
                l1_loss = torch.zeros([self.batch_size])
                if self.USE_CUDA:
                    advantage = advantage.cuda()
                    logprobs = logprobs.cuda()
                    logprobs2 = logprobs2.cuda()
                    acyclic_loss = acyclic_loss.cuda()
                    l1_loss = l1_loss.cuda()
                for i in range(self.batch_size):
                    for node in range(self.seq_len):
                        prob = probs[i,node,:].masked_select(actions_idxs[i, node, :])
                        # prob = probs[i,node,:]
                        prob_2 = probs[i,node,:].masked_select(~actions_idxs[i, node, :])
                        # if node == 0:
                            # print(prob)
                            # print(prob_2)
                            # print(actions_idxs[i, node, :], probs[i,node,:])
                        for prob_index in prob:
                            logprobs[i,node] += torch.log(prob_index)
                        for prob_index_2 in prob_2:
                            if prob_index_2 == 1:
                                print('!!!!!!!!!!!!!!!!!!')
                            logprobs[i,node] += torch.log(1-prob_index_2)

                    d = probs[i].shape[1]
                    acyclic_loss[i] = (trace_expm(probs[i]) - d)
                    
                    target = torch.zeros_like(probs[i])
                    l1_loss[i] = self.l1_loss_func(probs[i], target)
                # logprobs[logprobs < -1000] = -1000.  

                # if epoch == 0:
                #     actor_loss = l1_loss.mean() + acyclic_loss.mean()
                # else:
                # n5e9 100 True
                if self.global_score:
                    reinforce = -advantage.sum(1) * logprobs.sum(1)
                    if not self.mask:
                        actor_loss = reinforce.mean(0) + self.batch_size * self.acyclic_w * acyclic_loss.mean()
                    else:
                        actor_loss = reinforce.mean(0)
                else:
                    reinforce = -advantage * logprobs
                    if not self.mask:
                        actor_loss = reinforce.sum(1).mean(0)+ self.batch_size * self.acyclic_w * acyclic_loss.mean()
                    else:
                        actor_loss = reinforce.sum(1).mean(0)
                # actor_loss = reinforce.sum(1).mean(0)

                # if torch.isnan(actor_loss).sum() > 0:
                #     print("GGGGGGGGGGGGGGG")
                #     print(logprobs, advantage)
                #     exit(0)
                # with torch.autograd.detect_anomaly():
                # step = time.time()
                actor_loss.backward()
                # if is_nan(self.model.named_parameters()):
                #     print(reinforce)
                #     print("DDDDDDDDDDDDDDDD")

                torch.nn.utils.clip_grad_norm(self.model.actor.parameters(),
                                    float(self.max_grad_norm), norm_type=2)

                self.actor_optim.step()
                # if is_nan(self.model.named_parameters()):
                #     print("**************")

                # if R.mean(0).sum() > max_reward.sum():
                #     max_reward = R.mean(0).sum()
                #     max_actions = actions_idxs
                # max_reward = max_reward.detach()

                self.train_tour.append(R.data.sum(1).mean())
                self.losses.append(actor_loss.item())

                if (batch_id+1) % 100 == 0:
                    # print(R)
                    # print(rewards, mean_reward)

                    mean_losses = np.mean(np.abs(self.losses[-10:]))
                    # print(probs)
                    if store and batch_id >= 199:
                        store = False
                        torch.save(self.model.state_dict(), 'output/model/node'+ str(self.seq_len) + '_' + score + '.pth')
                        logger.debug("store model")
                    rewards = []
                    self.plot_file(self.epochs, filename)
                    # print("epoch: {}, batch id: {}, reward:{}, mean_losses:{}".format(epoch, batch_id, R.data.sum(1).mean(), mean_losses))
                    logger.debug("epoch: {}, batch id: {}, reward:{}, max reward:{} mean_losses:{} cyc:{}".format(epoch, batch_id, R.data.sum(1).mean(), max_reward.sum(), mean_losses, is_non_acyclic(actions_idxs[0])))
                # end = time.time()
                # print('total', end-start)
                # if batch_id == 2:
                #     break
                # if batch_id % 100 == 0:    

                #     self.model.eval()
                #     for val_batch in self.val_loader:
                #         inputs = Variable(val_batch)
                #         inputs = inputs.cuda()

                #         R, probs, actions_idxs = self.model(inputs)
                #         self.val_tour.append(R.data.mean())
                    if first == False and max_tmp == max_reward.sum():
                        count += 1
                    else:
                        max_tmp = max_reward.sum()
                        count = 0
                    if abs(mean_losses) < self.threshold or count > 3:
                        logger.info("EARLY STOPPAGE!")
                        # print(self.total_actions)
                        break      

            self.epochs += 1
        self.plot_file(self.epochs, filename)
        logger.info("max reward:{}".format(max_reward.sum()))
        logger.info("max actions:{}".format(max_actions))
        return max_actions
        
    def plot(self, epoch):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(1,2,1)
        plt.title('Score: epoch %s reward %s' % (epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        plt.subplot(1,2,2)
        plt.title('Score: epoch %s loss %s' % (epoch, self.losses[-1] if len(self.losses) else 'collecting'))
        plt.plot(self.losses)
        plt.grid()
        plt.show()
    
    def plot_file(self, epoch, filename):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(1,2,1)
        plt.title('Score: epoch %s reward %s' % (epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        plt.subplot(1,2,2)
        plt.title('Score: epoch %s loss %s' % (epoch, self.losses[-1] if len(self.losses) else 'collecting'))
        plt.plot(self.losses)
        plt.grid()
        plt.savefig("output/img/"+filename+'.jpg')