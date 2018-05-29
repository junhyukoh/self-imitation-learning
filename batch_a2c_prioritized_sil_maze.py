import os
import subprocess as sp

# 49 Atari games used in Nature DQN
envs = [
        'map0',
        'map1',
        ]

seeds = [0, 1, 2, 3, 4]
sil_updates = [0, 4]
sil_betas=[0.4]
count_exp_weights = [0, 1.0]

jobs = []
for seed in seeds:
    for env in envs:
        for sil_update in sil_updates:
            if sil_update > 0:
                 for sil_beta in sil_betas: 
                     for count_exp_weight in count_exp_weights:
                         log = '%s_s%d_c%d_b%g_exp%g' % (env,
                              seed,
                              sil_update,
                              sil_beta,
                              count_exp_weight)
                         jobs.append({
                              'env': env,
                              'seed': seed,
                              'sil_update': sil_update,
                              'sil_beta': sil_beta,
                              'count_exp_weight': count_exp_weight,
                              'log': log, 
                              })
            else:
                for count_exp_weight in count_exp_weights:
                    log = '%s_s%d_exp%g' % (env, seed, count_exp_weight)
                    jobs.append({
                        'env': env,
                        'seed': seed,
                        'sil_update': 0,
                        'sil_beta': 0,
                        'count_exp_weight': count_exp_weight,
                        'log': log, 
                        })

for job in jobs:
    print(job)

log_dir = 'result/a2c_prioritized_sil_maze'

sp.call(['mkdir', '-p', log_dir]) 
for job in jobs:
    path = os.path.join(log_dir, job['log'])
    if not os.path.exists(path):
        sp.call(['mkdir', path]) 
        print("Starting: ", job)
        sp.call(['python', 'baselines/a2c/run_maze_prioritized.py', 
            '--seed', str(job['seed']),
            '--env', str(job['env']),
            '--sil-update', str(job['sil_update']),
            '--sil-beta', str(job['sil_beta']),
            '--count-exp-weight', str(job['count_exp_weight']),
            '--log', log_dir + '/' + job['log']])
