import os
import subprocess as sp

# 49 Atari games used in Nature DQN
games = [
        'Alien',
        'Amidar',
        'Assault',
        'Asterix',
        'Asteroids',
        'Atlantis',
        'BankHeist',
        'BattleZone',
        'BeamRider',
        'Bowling',
        'Boxing',
        'Breakout',
        'Centipede',
        'ChopperCommand',
        'CrazyClimber',
        'DemonAttack',
        'DoubleDunk',
        'Enduro',
        'FishingDerby',
        'Freeway',
        'Frostbite',
        'Gopher',
        'Gravitar',
        'Hero',
        'IceHockey',
        'Jamesbond',
        'Kangaroo',
        'Krull',
        'KungFuMaster',
        'MontezumaRevenge',
        'MsPacman',
        'NameThisGame',
        'Pong',
        'PrivateEye',
        'Qbert',
        'Riverraid',
        'RoadRunner',
        'Robotank',
        'Seaquest',
        'SpaceInvaders',
        'StarGunner',
        'Tennis',
        'TimePilot',
        'Tutankham',
        'UpNDown',
        'Venture',
        'VideoPinball',
        'WizardOfWor',
        'Zaxxon',
        ]

log_dir = 'result/a2c_prioritized_sil'

seeds = [0,1,2]
sil_params = [
        [4, 0.4],    # updates, beta
        [0, 0.0]
        ]

jobs = []
for seed in seeds:
    for game in games:
        for (sil_update, sil_beta) in sil_params: 
            if sil_update > 0:
                log = '%s%d_c%d_b%g' % (game,
                        seed,
                        sil_update,
                        sil_beta)
                jobs.append({
                    'game': game,
                    'seed': seed,
                    'sil_update': sil_update,
                    'sil_beta': sil_beta,
                    'log': log, 
                    })
            else:
                log = '%s%d_a2c' % (game, seed)
                jobs.append({
                    'game': game,
                    'seed': seed,
                    'sil_update': 0,
                    'sil_beta': 0,
                    'log': log, 
                    })

for job in jobs:
    print(job)


sp.call(['mkdir', '-p', log_dir]) 
for job in jobs:
    path = os.path.join(log_dir, job['log'])
    if not os.path.exists(path):
        sp.call(['mkdir', path]) 
        print("Starting: ", job)
        sp.call(['python', 'baselines/a2c/run_atari_prioritized.py', 
            '--seed', str(job['seed']),
            '--env', job['game'] + 'NoFrameskip-v4',
            '--num-timesteps', str(int(50e6)),
            '--sil-update', str(job['sil_update']), 
            '--sil-beta', str(job['sil_beta']), 
            '--log', log_dir + '/' + job['log']])

