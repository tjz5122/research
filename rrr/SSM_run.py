import os
import shlex
import subprocess
import threading
import signal
from itertools import product

f = open("SSM_train_data", 'w')
f.close()

def run_command(env):
    print('=> run commands on GPU:{}', env['CUDA_VISIBLE_DEVICES'])

    while True:
        try:
            comm = command_list.pop(0)
        except IndexError:
            break

        proc = subprocess.Popen(comm, env=env)
        proc.wait()


test_param_groups = {}
test_param_groups["model_group"] = ['mgnet128','mgnet256','resnet18','resnet34']                 #SASA+: 2,4,8,16
test_param_groups["lr_group"] = [1.0]
test_param_groups["wd_group"] = [0.0005] #SASA+: 0.0005
test_param_groups["trail_group"] = [1]
test_param_groups["drop_factor_group"] = [10,15]           #SASA+: 2,5,10
test_param_groups["batch_size_group"] = [128]
test_param_groups["epochs_group"] = [150]
test_param_groups["leaky_group"] = [8,12]                 #SASA+: 2,4,8,16
test_param_groups["momentum_group"] = [0.9]
test_param_groups["dampening_group"] = [0, 0.9]
test_param_groups["significance_group"] = [0.05]          #SASA+ : 0.001,0.01,0.05,0.1
test_param_groups["samplefreq_group"] = [10,15]
test_param_groups["truncate_group"] = [0.02,0.03]
test_param_groups["ministate_group"] = [100]
test_param_groups["keymode_group"] = ["loss_plus_smooth"]
test_param_groups["varmode_group"] = ["bm"]
test_param_groups["data_group"] = ["cifar10"]



# flexible
value_list = [test_param_groups[group] for group in test_param_groups]
test_list = list(product(*value_list))
command = 'python SSM_train.py --cuda --model={} --lr={} --wd={} --trail={} --drop={} -b={} --epochs={} --lk={} -m={} --d={} --sig={} --sf={} --trun={} --minstat={} --km={} --vm={} --data={}'
# flexible

print("the total combinations of hyperparamater is", len(test_list))
command_list = []
for file in test_list:
    command_list += [command.format(*file)]
command_list = [shlex.split(comm) for comm in command_list]


# List all the GPUs you have

ids_cuda = [4,5,6,7]
for c in ids_cuda:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(c)
    thread = threading.Thread(target = run_command, args = (env, ))
    thread.start()