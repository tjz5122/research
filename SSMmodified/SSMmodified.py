import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from scipy import stats
import numpy as np
import torch.optim as optim
import torchvision
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SSM_Optimizer(Optimizer):


    def __init__(self, params, lr=-1, momentum=0, SSM_Optimizer_nu=1, weight_decay=0):
        # nu can take values outside of the interval [0,1], but no guarantee of convergence?
        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, SSM_Optimizer_nu=SSM_Optimizer_nu, weight_decay=weight_decay)
        super(SSM_Optimizer, self).__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.add_weight_decay()
        self.SSM_direction_and_update()
        return loss

    def add_weight_decay(self):
        # weight_decay is the same as adding L2 regularization
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay > 0:
                    p.grad.data.add_(weight_decay, p.data)

    def SSM_direction_and_update(self):

        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                #param_state = self.state[p]   # Optimization parameters
                x = p.data 
                g = p.grad.data
                state = self.state[p]
                # get momentum buffer.
                if 'momentum_buffer' not in state:
                    h = state['momentum_buffer'] = torch.zeros_like(x)
                else:
                    h = state['momentum_buffer']
                h.mul_(momentum).add_(1.0 - momentum, g)
                state['step_buffer'] = h
                #update
                p.data.add_(-group['lr'], state['step_buffer'])


                    
class Bucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0
        self.total_count = 0    # total number of elements added (including leaked)
        self.statistic = 0
 
    def reset(self):
        self.buffer.zero_()    
        self.count = 0          
        self.start = 0
        self.end = 0
        self.total_count = 0

    def double_size(self):
        self.size *= 2
        self.buffer.resize_(self.size)

    def add(self, val):
        if self.end == self.size:               # when the end index reach size
            self.double_size()                      # double the size of buffer

        self.buffer[self.end] = val             # always put new value at the end
        self.end += 1                           # and increase end index by one

        if self.fixed_len > 0:
            if self.count == self.fixed_len:
                self.start += 1
            else:
                self.count += 1
        else:
            if self.total_count % self.ratio == 0:  # if leaky_count is multiple of ratio
                self.count += 1                         # increase count in queue by one
            else:                                   # otherwise leak and keep same count
                self.start += 1                         # increase start index by one

        self.total_count += 1                   # always increase total_count by one

        # reset start index to 0 and end index to count to save space
        if self.start >= self.count:
            self.buffer[0:self.count] = self.buffer[self.start:self.end]
            self.start = 0
            self.end = self.count

    # ! Need to add safeguard to allow compute only if there are enough entries
    def mean_std(self, mode='bm'):
        mean = torch.mean(self.buffer[self.start:self.end]).item()

        if mode == 'bm':        # batch mean variance
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=b_n).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n /(len(Yks)-1))*torch.norm(diffs).item()
            dof = b_n - 1
        elif mode == 'olbm':    # overlapping batch mean
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=1).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n*self.count/(len(Yks)*(len(Yks)-1)))*torch.norm(diffs).item()
            dof = self.count - b_n
        else:                   # otherwise use mode == 'iid'
            std = torch.std(self.buffer[self.start:self.end]).item()
            dof = self.count - 1

        return mean, std, dof

    def stats_test(self, sigma, tolerance, mode='bm'):
        mean, std, dof = self.mean_std(mode=mode)

        # confidence interval
        t_sigma_dof = stats.t.ppf(1-sigma/2., dof)
        stat = std * t_sigma_dof / math.sqrt(self.count)
        self.statistic = stat
        
        return stat < tolerance


class SSM(SSM_Optimizer):

    def __init__(self, params, lr=-1, momentum=0, weight_decay=0, 
                 warmup=1000, drop_factor=10, significance=0.05, tolerance = 0.01, var_mode='mb',
                 leak_ratio=8, minN_stats=100, testfreq=100, samplefreq = 10, logstats=0, mode='loss with smooth'):

        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))
        if drop_factor < 1:
            raise ValueError("Invalid value for drop_factor (>=1): {}".format(drop_factor))
        if significance <= 0 or significance >= 1:
            raise ValueError("Invalid value for significance (0,1): {}".format(significance))
        if var_mode not in ['mb', 'olbm', 'iid']:
            raise ValueError("Invalid value for var_mode ('mb', 'olmb', or 'iid'): {}".format(var_mode))
        if leak_ratio < 1:
            raise ValueError("Invalid value for leak_ratio (int, >=1): {}".format(leak_ratio))
        # if minN_stats < 100:
        #     raise ValueError("Invalid value for minN_stats (int, >=100): {}".format(minN_stats))
        if warmup < 0:
            raise ValueError("Invalid value for warmup (int, >1): {}".format(warmup))
        if testfreq < 1:
            raise ValueError("Invalid value for testfreq (int, >=1): {}".format(testfreq))

        super(SSM, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        # New Python3 way to call super()
        # super().__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)

        # State initialization: leaky bucket belongs to global state.
        p = self.param_groups[0]['params'][0]
        if 'bucket' not in self.state:
            self.state['bucket'] = Bucket(1000, leak_ratio, p.dtype, p.device)

        self.state['lr'] = float(lr)
        self.state['momemtum'] = float(momentum)
        self.state['drop_factor'] = drop_factor
        self.state['significance'] = significance
        self.state['tolerance'] = tolerance
        self.state['var_mode'] = var_mode
        self.state['minN_stats'] = int(minN_stats)
        self.state['warmup'] = int(warmup)
        self.state['samplefreq'] = int(samplefreq)
        self.state['testfreq'] = int(testfreq)
        self.state['logstats'] = int(logstats)
        self.state['nSteps'] = 0
        self.state['loss'] = 0
        self.state['mode'] = mode


        # statistics to monitor
        self.state['smoothing'] = 0
        self.state['stats_x1d'] = 0
        self.state['stats_ld2'] = 0
        self.state['stats_val'] = 0
        self.state['stats_test'] = 0
        self.state['statistic'] = 0
        self.state['stats_stationary'] = 0
        self.state['stats_mean'] = 0


    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates model and returns loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.add_weight_decay()
        self.SSM_direction_and_update()
        self.state['nSteps'] += 1
        self.stats_adaptation()

        return loss

    def stats_adaptation(self):
        
        
        bucket = self.state['bucket']
        # add the statistic in to the bucket according to the frequency 
        if self.state['nSteps'] % self.state['samplefreq'] == 0:

            if self.state['mode'] == 'loss with smooth':
                fristpart = 0
                secondpart = 0
        
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        xk1 = p.data.view(-1)
                        dk = self.state[p]['step_buffer'].data.view(-1)     # OK after super().step()
                        fristpart += xk1.dot(dk).item()
                        secondpart += dk.dot(dk).item()
                secondpart *= 0.5 * self.state['lr']
        
                '''
                dk = self._gather_flat_buffer('step_buffer')
                xk1 = self._gather_flat_param() 
                gk = self._gather_flat_grad()
                '''
                
                #self.state['smoothing'] = xk1.dot(gk).item() + (0.5 * self.state['lr']) * ((1 + self.state['momemtum'])/(1 - self.state['momemtum'])) * (dk.dot(dk).item())
                self.state['smoothing'] = fristpart + secondpart
                self.state['stats_val'] =  self.state['loss'] + self.state['smoothing']
        
            if self.state['mode'] == 'loss':
                self.state['stats_val'] =  self.state['loss'] 
            
            if self.state['mode'] == 'sasa_plus':
                self.state['stats_x1d'] = xk1.dot(dk).item()
                self.state['stats_ld2'] = (0.5 * self.state['lr']) * (dk.dot(dk).item())
                self.state['stats_val'] = self.state['stats_x1d'] + self.state['stats_ld2']
        
            ## add statistic to leaky bucket
            bucket.add(self.state['stats_val'])
        
        
        # check statistics and adjust learning rate
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        self.state['statistic'] = 0
        
        #if the bucket length is large enough and it is time to do the test (generally is when epoch ends)
        if (bucket.count > self.state['minN_stats']) and (self.state['nSteps'] % self.state['testfreq'] == 0):
            stationary= bucket.stats_test(self.state['significance'], self.state['tolerance'], self.state['var_mode'])
            self.state['stats_test'] = 1
            self.state['statistic'] = bucket.statistic
            self.state['stats_stationary'] = int(stationary)
    
            ## perform statistical test for stationarity
            if self.state['warmup'] < self.state['nSteps'] and self.state['stats_stationary'] == 1:
                self.state['lr'] /= self.state['drop_factor']
                for group in self.param_groups:
                    group['lr'] = self.state['lr']
                self._zero_buffers('momentum_buffer') #????
                bucket.reset()


    def _gather_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    view = p.data.new(p.data.numel()).zero_()
                elif p.grad.data.is_sparse:
                    view = p.grad.data.to_dense().view(-1) #take derivative of p
                else:
                    view = p.grad.data.view(-1) 
                views.append(view)
        return torch.cat(views, 0)
    
    # methods for gather flat parameters
    def _gather_flat_param(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    # method for gathering/initializing flat buffers that are the same shape as the parameters
    def _gather_flat_buffer(self, buf_name):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name not in state:  # init buffer
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = state[buf_name].data.view(-1)
                views.append(view)
        return torch.cat(views, 0) 
    
    
    def _zero_buffers(self, buf_name):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name in state:
                    state[buf_name].zero_()
        return None
