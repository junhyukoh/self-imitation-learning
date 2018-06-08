# Introduction
This repository is an implementation of [ICML 2018 Self-Imitation Learning](https://arxiv.org/abs/1707.03497) in Tensorflow.
```
@inproceedings{Oh2018SIL,
  title={Self-Imitation Learning},
  author={Junhyuk Oh and Yijie Guo and Satinder Singh and Honglak Lee},
  booktitle={ICML},
  year={2018}
}
```
Our code is based on [OpenAI Baselines](https://github.com/openai/baselines).

# Training
The following command runs `A2C+SIL` on Atari games:
```
python baselines/a2c/run_atari_sil.py --env FreewayNoFrameskip-v4
```

The following command runs `PPO+SIL` on MuJoCo tasks:
```
python baselines/ppo2/run_mujoco_sil.py --env Ant-v2 --num-timesteps 10000000 --lr 5e-05
```
