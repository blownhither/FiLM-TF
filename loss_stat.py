import glob
import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('log_name', default=None)
args = parser.parse_args()

log_name = args.log_name
if log_name is None:
    log_name = sorted(glob.glob('KerPred-*.log'))[-1]
print(log_name)

output = subprocess.check_output(f'cat {log_name} | grep "eval loss= "',
                                 shell=True)
output = [x for x in output.decode().split('\n') if len(x)]
print(len(output), 'records')
acc = [float(x.split('acc=')[1].strip()) for x in output]
loss = [float(x.split('acc=')[0].split('loss=')[1].strip()) for x in output]


def avg_half(a):
    return np.mean(a[len(a)//2:])


def window_min(a, win_size=None):
    if win_size is None:
        win_size = max(3, len(a) // 20)     # 5% training length window
    a = np.convolve(a, np.ones(win_size) / win_size, mode='valid')
    return min(a)


def last_window(a, win_size=None):
    if win_size is None:
        win_size = max(3, len(a) // 20)     # 5% training length window
    win_size = min(win_size, len(a))
    return np.mean(a[-win_size:])


print('acc | loss')
print('half', avg_half(acc), avg_half(loss))
print('win', window_min(acc), window_min(loss))
print('last', last_window(acc), last_window(loss))
print('extreme', np.max(acc), np.min(loss))
