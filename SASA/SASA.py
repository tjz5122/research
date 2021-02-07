import torch
import torch.nn.functional as F
import math
from scipy import stats
from collections import defaultdict, Iterable
from copy import deepcopy
from itertools import chain
from tensorboardX import SummaryWriter

# Added by Lian
required = object()

# Added by Lian
class Optimizer(object):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save ids instead of Tensors
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups] #return a list that contains dicts
        # Remap state to use ids as keys
        packed_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

# keep the most recent half of the added items
class HalfQueue(object):
    def __init__(self, maxN, like_tens):
        self.q = torch.zeros(maxN, dtype=like_tens.dtype,
                             device=like_tens.device)
        self.n = 0 #number filled
        self.remove = False  
        self.maxN = maxN

    def double(self):
        newqueue = torch.zeros(self.maxN * 2, dtype=self.q.dtype,
                               device=self.q.device)
        newqueue[0:self.maxN][:] = self.q
        self.q = newqueue
        self.maxN *= 2

    def add(self, val):   #remove 2 element when add one element to get 1/2 of the list
        if self.remove is True:
            # remove 1
            self.q[:-1] = deepcopy(self.q[1:]) # probably slow but ok for now
        else:
            self.n += 1  #add one
        self.q[self.n - 1] = val
        if self.n == self.maxN:
            self.double()
        self.remove = not self.remove  # or self.n == self.maxN)

    def mean_std(self, mode='bm'): #default algorithm is batch mean variance
        gbar = torch.mean(self.q[:self.n])  #gbar = z_bar_n = mean for the all z 
        std_dict = {} #standard deviation 
        df_dict = {}  #degree of freedom

        # sample variance for iid samples.
        std = torch.std(self.q[:self.n]) #sample variance for all z
        std_dict['iid'] = std
        df_dict['iid'] = self.n - 1

        # batch mean variance
        b_n = int(math.floor(math.sqrt(self.n)))   #a_n = len(Yks) / b_n(batch number) take n^0.5
        Yks = F.avg_pool1d(self.q[:self.n].unsqueeze(0).unsqueeze(0),
                           kernel_size=b_n, stride=b_n).view(-1) # Yks = bar_y_s(batch mean)
        
        diffs = Yks - gbar   # bar_y_s - bar_z_n
        std = math.sqrt(b_n / (len(Yks) - 1)) * torch.norm(diffs) # hat_theta_BM = variance of batchs
        std_dict['bm'] = std
        df_dict['bm'] = b_n - 1

        # overlapping batch mean
        Yks = F.avg_pool1d(self.q[:self.n].unsqueeze(0).unsqueeze(0),
                            kernel_size=b_n, stride=1).view(-1)
        diffs = Yks - gbar
        std = math.sqrt(
            b_n * self.n / (len(Yks) * (len(Yks) - 1))) * torch.norm(diffs)
        std_dict['olbm'] = std
        df_dict['olbm'] = self.n - b_n

        return gbar, std_dict[mode], df_dict[mode]#, std_dict, df_dict
        # total mean / variance of batchs / degree of freedom of batches

    def reset(self):
        self.n = 0
        self.remove = False
        self.q.zero_()



# returns True if |u-v| < delta*u with signif level sigma.
def test_onesamp(z, v, sigma, delta, mode='bm', verbose=True):
    z_mean, z_std, z_df, stds, dfs = z.mean_std(mode=mode)
    v_mean, _, _, _, _ = v.mean_std()

    rhs = delta * v_mean

    K = z.n  # number of samples

    t_sigma_df = stats.t.ppf(1 - sigma / 2., z_df)
    z_upperbound = z_mean + z_std.mul(t_sigma_df / math.sqrt(K))
    z_lowerbound = z_mean - z_std.mul(t_sigma_df / math.sqrt(K))


    return (z_upperbound < rhs and z_lowerbound > -rhs), z_mean, z_upperbound, z_lowerbound, rhs, stds, dfs


class SASA(Optimizer):

    def step_onesamp(self, closure=None):
        # assert len(self.param_groups) == 1 # same as lbfgs
        # before gathering the gradient, add weight decay term
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            if weight_decay != 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data.add_(weight_decay, p.data)

        group = self.param_groups[0]
        minN = group['minN']
        maxN = group['maxN']
        zeta = group['zeta']   #decrease rate of lr (lr = lr *zeta)
        momentum = group['momentum']  # beta
        delta = group['delta'] # test if |bar_z_n / bar_v_n| < delta
        sigma = group['sigma']  #confidence level

        # like in LBFGS, set global state to be state of first param.
        state = self.state[self._params[0]]


        if len(state) == 0:
            state['step'] = 0
            state['K'] = 0  # how many samples we have
            state['z'] = HalfQueue(maxN, self._params[0])
            state['v'] = HalfQueue(maxN, self._params[0])


        g_k = self._gather_flat_grad()   #tensor
        x_k = self._gather_flat_param()    #tensor  #theta(parameter)
        d = self._gather_flat_buf('momentum_buffer')  # d is d in pdf/return a tensor

        uk = g_k.dot(x_k)   # u_k is <x^k, g^k>
        vk = d.dot(d).mul(
            0.5 * group['lr'] * (1.0 + momentum) / (1.0 - momentum)) # v_k is a/2 * (1+b)/(1-b) E[<d,d>]

        state['z'].add(uk - vk)  # define z_k
        state['v'].add(vk)

        if closure is not None:
            u = uk.item()  # return one value
            v = vk.item()  
            z = u - v
            closure([u], [v], [z], [], [], [], [], [])
        
        
        self.lowercriteria = 0
        self.uppercriteria = 0
        self.tolerance = 0

        if state['K'] >= minN and state['K'] % group['testfreq'] == 0:
            u_equals_v, z_mean, z_upperbound, z_lowerbound, rhs, stds, dfs = test_onesamp(
                state['z'], state['v'], sigma, delta, mode=self.mode)
            self.lowercriteria = z_lowerbound
            self.uppercriteria = z_upperbound
            self.tolerance = rhs
            
            if closure is not None:
                closure([], [], [], [z_mean.item()], [z_upperbound.item()], [z_lowerbound.item()], [rhs.item()],
                        stds, dfs)
            if state['step'] > self.warmup and u_equals_v:

                group['lr'] = group['lr'] * zeta  # decrease lr by zeta 
                state['K'] = 0  # need to collect at least minN more samples.
                # should reset the queues here; bad if samples from before corrupt what you have now.
                state['z'].reset()
                state['v'].reset()
        elif self.logstats:
            if state['z'].n >= 4 and state['K'] % self.logstats == 0:
                u_equals_v, z_mean, z_upperbound, z_lowerbound, rhs, stds, dfs = test_onesamp(
                    state['z'], state['v'], sigma, delta, mode=self.mode,
                    verbose=False)
                if closure is not None:
                    closure([], [], [], [z_mean.item()], [z_upperbound.item()], [z_lowerbound.item()],
                            [rhs.item()], stds, dfs)

        state['K'] += 1

        for p in self._params:
            if p.grad is None:
                continue
            param_state = self.state[p]
            g_k = p.grad.data
            # get momentum buffer.
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(1.0 - momentum, g_k)
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1.0 - momentum, g_k)

            # now do update.
            p.data.add_(-group['lr'], buf)

        state['step'] += 1

    def __init__(self, params, lr=required, weight_decay=0, momentum=0,
                 warmup=0, minN=100, maxN=1000, zeta=0.1, sigma=0.2, delta=0.02,
                 testfreq=500, onesamp=True, mode='bm', logstats=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, minN=minN, maxN=maxN,
                        zeta=zeta, momentum=momentum, sigma=sigma, delta=delta,
                        testfreq=testfreq)

        super(SASA, self).__init__(params, defaults)
        if onesamp:
            self.step_fn = self.step_onesamp
        else:
            self.step_fn = self.step_twosamp
        # self._params = self.param_groups[0]['params']
        self._params = []
        for param_group in self.param_groups:
            self._params += param_group['params']
        self.warmup = warmup  # todo: warmup in state?
        self.mode = mode  # using which variance estimator
        self.lowercriteria = 0
        self.uppercriteria = 0
        self.tolerance = 0
        print("using variance estimator: ", mode)
        self.logstats = logstats
        print("logging stats every {} steps".format(logstats))

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1) 
            else:
                view = p.grad.data.view(-1) 
            views.append(view)
        return torch.cat(views, 0) 


    def _gather_flat_buf(self, buf_name):
        views = []
        for p in self._params:
            param_state = self.state[p]
            if buf_name not in param_state:  # init buffer
                view = p.data.new(p.data.numel()).zero_()  
            else:
                view = param_state[buf_name].data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_param(self):
        views = []
        for p in self._params:
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def step(self, closure=None):
        self.step_fn(closure=closure)

