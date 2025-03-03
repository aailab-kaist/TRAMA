# TRAMA
(Official) PyTorch implementation for Trajectory-Class-Aware Multi-Agent Reinforcement Learning (ICLR 2025)

# Note
This codebase accompanies the paper submission "**Trajectory-Class-Aware Multi-agent Reinforcement Learning (TRAMA)**" and is based on [PyMARL](https://github.com/oxwhirl/pymarl), [SMAC](https://github.com/oxwhirl/smac), and [SMAC2](https://github.com/oxwhirl/smacv2) which are open-sourced.
The paper is accepted by [ICLR2025](https://iclr.cc/Conferences/2025/) and now available in [OpenReview](https://openreview.net/forum?id=uqe5HkjbT9).

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning and our code includes implementations of the following algorithm:
- [**LAGMA**: LAtent Goal-guided Multi-Agent Reinforcement Learning ](https://arxiv.org/abs/2405.19998)

# Run an experiment
To train TRAMA on surComb3 in SC2(v2), run the following command:
```
python3 src/main.py --config=trama_gc_qplex --env-config=sc2_gen_protoss_surComb3
```

To train TRAMA on sc2_gen_protoss in SC2(v2), run the following command:
```
python3 src/main.py --config=trama_gc_qplex --env-config=sc2_gen_protoss
```

# Publication
If you find this repository useful, please cite our paper:
```
@inproceedings{na2025trama,
  title={Trajectory-class-aware Multi-agent Reinforcement Learning},
  author={Na, Hyungho and Lee, Kwanghyeon and Lee, Sumin and Moon, Il-chul},
  journal={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
