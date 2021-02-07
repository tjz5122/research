#@title SASA+ tianhao
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:27:06 2021

@author: thzha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:50:23 2020

@author: thzha
"""
import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from scipy import stats

class QHM(Optimizer):
    r"""
    Stochastic gradient method with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k)
        x(k+1) = x(k) - \alpha * d(k)
    "Quasi-hyperbolic momentum and Adam for deep learning"
        by Jerry Ma and Denis Yarats, ICLR 2019
    optimizer = QHM(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0)
    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range[0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range[0,1] (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum \beta and dampened gradient (1-\beta)
            \nu = \beta: SGD with "Nesterov momentum" \beta
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    Example:
        >>> optimizer = torch.optim.QHM(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0):
        # nu can take values outside of the interval [0,1], but no guarantee of convergence?
        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, qhm_nu=qhm_nu, weight_decay=weight_decay)
        super(QHM, self).__init__(params, defaults)

        # extra_buffer == True only in SSLS with momentum > 0 and nu != 1
        self.state['allocate_step_buffer'] = False

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
        self.qhm_direction()
        self.qhm_update()

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

    def qhm_direction(self):

        for group in self.param_groups:
            momentum = group['momentum']
            qhm_nu = group['qhm_nu']

            for p in group['params']:
                if p.grad is None:
                    continue
                x = p.data  # Optimization parameters
                g = p.grad.data  # Stochastic gradient

                # Compute the (negative) step directoin d and necessary momentum
                state = self.state[p]
                if abs(momentum) < 1e-12 or abs(qhm_nu) < 1e-12:  # simply SGD if beta=0 or nu=0
                    d = state['step_buffer'] = g
                else:
                    if 'momentum_buffer' not in state:
                        h = state['momentum_buffer'] = torch.zeros_like(x)
                    else:
                        h = state['momentum_buffer']
                    # Update momentum buffer: h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
                    h.mul_(momentum).add_(1 - momentum, g)

                    if abs(qhm_nu - 1) < 1e-12:  # if nu=1, then same as SGD with momentum
                        d = state['step_buffer'] = h
                    else:
                        if self.state['allocate_step_buffer']:  # copy from gradient
                            if 'step_buffer' not in state:
                                state['step_buffer'] = torch.zeros_like(g)
                            d = state['step_buffer'].copy_(g)
                        else:  # use gradient buffer
                            d = state['step_buffer'] = g
                        # Compute QHM momentum: d(k) = (1 - \nu) * g(k) + \nu * h(k)
                        d.mul_(1 - qhm_nu).add_(qhm_nu, h)

    def qhm_update(self):
        """
        Perform QHM update, need to call compute_qhm_direction() before calling this.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-group['lr'], self.state[p]['step_buffer'])
                    
                    
class LeakyBucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        '''
        size:  size of allocated memory buffer to keep the leaky bucket queue,
               which will be doubled whenever the memory is full
        ratio: integer ratio of total number of samples to numbers to be kept:
               1 - keep all, 
               2 - keep most recent 1/2, 
               3 - keep most recent 1/3,
               ... 
        fixed_len: fixed length to keep, ratio >=1 becomes irrelevant
        '''
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0
        self.total_count = 0    # total number of elements added (including leaked)
        self.lowCriteria = 0
        self.highCriteria = 0
        self.meanCriteria = 0
 
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

    def stats_test(self, sigma, mode='bm', composite_test=False):
        mean, std, dof = self.mean_std(mode=mode)

        # confidence interval
        t_sigma_dof = stats.t.ppf(1-sigma/2., dof)
        half_width = std * t_sigma_dof / math.sqrt(self.count)
        lower = mean - half_width
        upper = mean + half_width
        # The simple confidence interval test    
        # stationarity = lower < 0 and upper > 0

        # A more stable test is to also check if two half-means are of the same sign
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        self.lowCriteria = lower
        self.highCriteria = upper
        self.meanCriteria = mean1 * mean2
        
        stationarity = (lower < 0 and upper > 0) and (mean1 * mean2 > 0)
        

        if composite_test:
            # Use two half tests to avoid false positive caused by crossing 0 in transient phase
            lb1 = mean1 - half_width
            ub1 = mean1 + half_width
            lb2 = mean2 - half_width
            ub2 = mean2 + half_width
            self.lowCriteria = lb1 * ub1
            self.highCriteria = lb2 * ub2
            self.meanCriteria = mean1 * mean2
            stationarity = (lb1 * ub1 < 0) and (lb2 * ub2 < 0) and (mean1 * mean2 > 0)

        return stationarity, mean, lower, upper  

    # method to test if average loss after line search is no longer decreasing 
    def rel_reduction(self):
        if self.count < 4:
            return 0.5
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        return (mean1 - mean2) / mean1
        
    # method to test if average loss after line search is no longer decreasing 
    def is_decreasing(self, min_cnt=1000, dec_rate=0.01):
        if self.count < min_cnt:
            return True
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        return (mean1 - mean2) / mean1 > dec_rate

    def linregress(self, sigma, mode='linear'):
        """
        calculate a linear regression
        sigma: the confidence of the one-side test
            H0: slope >= 0 vs H1: slope < 0
        mode: whether log scale the x axis
        """
        TINY = 1.0e-20
        x = torch.arange(self.total_count-self.count, self.total_count,
                         dtype=self.buffer.dtype, device=self.buffer.device)
        if mode == 'log':
            x = torch.log(x)
        # both x and y has dimension (self.count,)
        xy = torch.cat([x.view(1, -1),
                        self.buffer[self.start:self.end].view(1, -1)],
                       dim=0)
        # compute covariance matrix
        fact = 1.0 / self.count
        xy -= torch.mean(xy, dim=1, keepdim=True)
        xyt = xy.t()
        cov = fact * xy.matmul(xyt).squeeze()
        # compute the t-statistics
        r_num = cov[0, 1].item()
        r_den = torch.sqrt(cov[0, 0]*cov[1, 1]).item()
        if r_den == 0.0:
            r = 0.0
        else:
            r = r_num / r_den
            # test for numerical error propagation
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0

        df = self.count - 2
        t = r * math.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        # one-sided test for decreasing
        prob = stats.t.cdf(t, df)
        is_decreasing = prob < sigma
        # slop
        slope = r_num / cov[0, 0].item()
        return is_decreasing, slope, prob
    

class SASA(QHM):
    r"""
    Statistical Adaptive Stochastic Approximation (SASA+) with master condition.
    optimizer = SASA(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0, 
                     warmup=0, drop_factor=2, significance=0.02, var_mode='bm', 
                     leak_ratio=4, minN_stats=400, testfreq=100, logstats=0)
    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha * d(k)   
    Stationary criterion: 
        E[ <x(k),   d(k)>] - (\alpha / 2) * ||d(k)||^2 ] = 0
    or equivalently,
        E[ <x(k+1), d(k)>] + (\alpha / 2) * ||d(k)||^2 ] = 0
    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range(0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range(0,1) (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum and dampened gradient
            \nu = \beta: SGD with "Nesterov momentum"
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        warmup (int, optional): number of steps before testing (default: 100)
        dropfactor (float, optional): factor of drop learning rate (default: 10)
        significance (float, optional): test significance level (default:0.05)  
        var_mode (string, optional): variance computing mode (default: 'mb')
        leak_ratio (int, optional): leaky bucket ratio to kept (default: 8)
        minN_stats (int, optional): min number of samples for test (default: 1000)
        testfreq (int, optional): number of steps between testing (default:100)
        logstats (int, optional): number of steps between logs (0 means no log)
    Example:
        >>> optimizer = torch.optim.SASA(model.parameters(), lr=0.1, momentum=0.9, 
        >>>                              weight_decay=0.0005)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0, 
                 warmup=1000, drop_factor=10, significance=0.05, var_mode='mb',
                 leak_ratio=8, minN_stats=1000, testfreq=100, logstats=0):

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
        if minN_stats < 100:
            raise ValueError("Invalid value for minN_stats (int, >=100): {}".format(minN_stats))
        if warmup < 0:
            raise ValueError("Invalid value for warmup (int, >1): {}".format(warmup))
        if testfreq < 1:
            raise ValueError("Invalid value for testfreq (int, >=1): {}".format(testfreq))

        super(SASA, self).__init__(params, lr=lr, momentum=momentum, qhm_nu=qhm_nu, weight_decay=weight_decay)
        # New Python3 way to call super()
        # super().__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)

        # State initialization: leaky bucket belongs to global state.
        p = self.param_groups[0]['params'][0]
        if 'bucket' not in self.state:
            self.state['bucket'] = LeakyBucket(1000, leak_ratio, p.dtype, p.device)

        self.state['lr'] = float(lr)
        self.state['drop_factor'] = drop_factor
        self.state['significance'] = significance
        self.state['var_mode'] = var_mode
        self.state['minN_stats'] = int(minN_stats)
        self.state['warmup'] = int(warmup)
        self.state['testfreq'] = int(testfreq)
        self.state['logstats'] = int(logstats)
        self.state['composite_test'] = True     # first drop use composite test
        self.state['nSteps'] = 0                # steps counter +1 every iteration

        # statistics to monitor
        self.state['stats_x1d'] = 0
        self.state['stats_ld2'] = 0
        self.state['stats_val'] = 0
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        self.state['stats_mean'] = 0
        self.state['stats_lb'] = 0
        self.state['stats_ub'] = 0
        self.state["lowCriteria"] = 0
        self.state['highCriteria'] = 0
        self.state['meanCriteria'] = 0

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
        self.qhm_direction()
        self.qhm_update()
        self.state['nSteps'] += 1
        self.stats_adaptation()

        return loss

    def stats_adaptation(self):

        # compute <x(k+1), d(k)> and ||d(k)||^2 for statistical test
        self.state['stats_x1d'] = 0.0
        self.state['stats_ld2'] = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                xk1 = p.data.view(-1)
                dk = self.state[p]['step_buffer'].data.view(-1)     # OK after super().step()
                self.state['stats_x1d'] += xk1.dot(dk).item()
                self.state['stats_ld2'] += dk.dot(dk).item()
        self.state['stats_ld2'] *= 0.5 * self.state['lr']

        # Gather flat buffers can take too much memory for large models
        # Compute <x(k+1), d(k)> and ||d(k)||^2 for statistical test
        # dk = self._gather_flat_buffer('step_buffer')
        # xk1 = self._gather_flat_param() 
        # self.state['stats_x1d'] = xk1.dot(dk).item()
        # self.state['stats_ld2'] = (0.5 * self.state['lr']) * (dk.dot(dk).item())

        # add statistic to leaky bucket
        self.state['stats_val'] = self.state['stats_x1d'] + self.state['stats_ld2']
        bucket = self.state['bucket']
        bucket.add(self.state['stats_val'])

        # check statistics and adjust learning rate
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        self.state['lowCriteria'] = 0
        self.state['highCriteria'] = 0
        self.state['meanCriteria'] = 0
        if bucket.count > self.state['minN_stats'] and self.state['nSteps'] % self.state['testfreq'] == 0:
            stationary, mean, lb, ub = bucket.stats_test(self.state['significance'], 
                                                     self.state['var_mode'], 
                                                     self.state['composite_test'])
            self.state['stats_test'] = 1
            self.state['stats_stationary'] = int(stationary)
            
            #Microsoft stat
            self.state['stats_mean'] = mean
            self.state['stats_lb'] = lb
            self.state['stats_ub'] = ub
            #Tianhao added
            self.state['lowCriteria'] = self.state['bucket'].lowCriteria
            self.state['highCriteria'] = self.state['bucket'].highCriteria
            self.state['meanCriteria'] = self.state['bucket'].meanCriteria
            
            # perform statistical test for stationarity
            if self.state['nSteps'] > self.state['warmup'] and self.state['stats_stationary'] == 1:
                self.state['lr'] /= self.state['drop_factor']
                for group in self.param_groups:
                    group['lr'] = self.state['lr']
                self._zero_buffers('momentum_buffer')
                self.state['composite_test'] = False
                bucket.reset()

        # Log statistics only for debugging. Therefore self.state['stats_test'] remains False     
        if self.state['logstats'] and not self.state['stats_test']:
            if bucket.count > bucket.ratio and self.state['nSteps'] % self.state['logstats'] == 0:
                stationary, mean, lb, ub = bucket.stats_test(self.state['significance'], 
                                                              self.state['var_mode'],
                                                              self.state['composite_test'])
                self.state['stats_stationary'] = int(stationary)
                self.state['stats_mean'] = mean
                self.state['stats_lb'] = lb
                self.state['stats_ub'] = ub


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