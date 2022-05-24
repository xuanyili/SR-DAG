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

class ValidModel:
    def __init__(self, model, valid_dataset, batch_size=10, threshold=0.2, max_grad_norm=2., USE_CUDA = True):
        self.model = model
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.seq_len = valid_dataset.getnodesize()
        
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.actor_optim   = optim.Adam(model.actor.parameters(), lr=1e-4)
        self.max_grad_norm = max_grad_norm
        
        self.valid_dag = []
        self.val_dag   = []
        self.losses = []
        self.USE_CUDA = USE_CUDA
        self.epochs = 0
        # torch.autograd.set_detect_anomaly(True)
    
    def valid_and_validate(self, n_epochs, logger, filename='none'):
        max_reward = torch.zeros(1)
        max_tmp = torch.zeros(1)
        mean_reward = torch.zeros(1)
        rewards = []
        losses =[]
        first = True
        count = 0

        if self.USE_CUDA: 
            max_reward = max_reward.cuda()
            mean_reward = mean_reward.cuda()
        self.model.train()
        for epoch in range(n_epochs):
            for batch_id, sample_batch in enumerate(self.valid_loader):
                # start = time.time()
                self.actor_optim.zero_grad()
                inputs = sample_batch
                if self.USE_CUDA:
                    inputs = inputs.cuda()
                # print(inputs)
                R, probs, probs_2, actions_idxs, mask = self.model(inputs)
                # mean_reward = R.mean(0)
                rewards.append(R)
                mean_reward = torch.cat(rewards).mean(0)
                sum_reward = R.sum(1)
                _, sorted_reward_idx = sum_reward.sort(descending=True)
                for idx in sorted_reward_idx:
                    if is_non_acyclic(actions_idxs[idx]) < 1e-10:
                        max_reward_idx = idx
                        if first:
                            first = False

                            max_reward = R[max_reward_idx]
                            max_actions = actions_idxs[max_reward_idx]
                        else:
                            if sum_reward[max_reward_idx] > max_reward.sum():
                                max_reward = R[max_reward_idx]
                                max_actions = actions_idxs[max_reward_idx]
                        break
                # else:
                #     critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean(0))
                if len(max_reward) == 1:
                    advantage = (R - mean_reward)
                else:
                    advantage = (R - max_reward)
                # advantage = (first_reward - R)

                logprobs = torch.zeros([self.batch_size, self.seq_len])
                logprobs2 = torch.zeros([self.batch_size, self.seq_len])
                acyclic_loss = torch.zeros([self.batch_size, self.seq_len])
                if self.USE_CUDA:
                    advantage = advantage.cuda()
                    logprobs = logprobs.cuda()
                    logprobs2 = logprobs2.cuda()
                    acyclic_loss = acyclic_loss.cuda()
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

                reinforce = -advantage * logprobs
                actor_loss = reinforce.sum(1).mean(0) + self.batch_size * 500 * acyclic_loss.mean()

                # if torch.isnan(actor_loss).sum() > 0:
                #     print("GGGGGGGGGGGGGGG")
                #     print(logprobs, advantage)
                #     exit(0)
                # with torch.autograd.detect_anomaly():
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

                self.valid_dag.append(R.data.sum(1).mean())
                self.losses.append(actor_loss.item())

                if (batch_id+1) % 100 == 0:
                    # print(R)
                    # print(rewards, mean_reward)
                    # learn_rate = learn_rate + 5
                    mean_losses = np.mean(np.abs(self.losses[-10:]))
                    # print(probs)
                    rewards = []
                    self.plot_file(self.epochs, filename)
                    # print("epoch: {}, batch id: {}, reward:{}, mean_losses:{}".format(epoch, batch_id, R.data.sum(1).mean(), mean_losses))
                    logger.debug("epoch: {}, batch id: {}, reward:{}, max reward:{} mean_losses:{} cyc:{}".format(epoch, batch_id, R.data.sum(1).mean(), max_reward.sum(), mean_losses,is_non_acyclic(actions_idxs[0])))
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
                #         self.val_dag.append(R.data.mean())
            
                    if max_tmp == max_reward.sum():
                        count += 1
                    else:
                        max_tmp = max_reward.sum()
                        count = 0
                    if abs(mean_losses) < self.threshold or count > 3:
                        logger.info("EARLY STOPPAGE!")
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
        plt.title('Score: epoch %s reward %s' % (epoch, self.valid_dag[-1] if len(self.valid_dag) else 'collecting'))
        plt.plot(self.valid_dag)
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
        plt.title('Score: epoch %s reward %s' % (epoch, self.valid_dag[-1] if len(self.valid_dag) else 'collecting'))
        plt.plot(self.valid_dag)
        plt.grid()
        plt.subplot(1,2,2)
        plt.title('Score: epoch %s loss %s' % (epoch, self.losses[-1] if len(self.losses) else 'collecting'))
        plt.plot(self.losses)
        plt.grid()
        plt.savefig("output/img/"+filename+'.jpg')