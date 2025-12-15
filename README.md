# Alpha RL Throw 🚀

Refer to the CONDOR-EPIC: [CDR-1586](https://theaiinstitute.atlassian.net/browse/CDR-1586)

This is a forked setup from [GCR-PPO](https://github.com/humphreymunn/GCR-PPO), a focused fork/adaptation of [IsaacLab](https://github.com/isaac-sim/IsaacLab) targeting **IsaacLab 2.1.0** with **RSL-RL**.

Relevant background 🎯:
- 🔧 Optimization-only throwing on fixed-base, single-arm manipulators.
- 🤖 Recent RL work on throwing with single-arm manipulators on legged bases (Anymal/Spot).
- 📐 Some slanted-throw approaches rely on real-robot data for sim2real and imitation.
- 🧩 Open-source code for multi-objective throwing remains scarce.

GCR-PPO, a PPO variant, includes a throwing task on a G1 humanoid from prior work. We reuse it here to recreate throwing behavior with observations, actions, and rewards adapted to the RAI Alpha robot. Any PPO algorithm could be swapped in.


## Contents
- [Alpha RL Throw](#alpha-rl-throw-)
- [Quick Start](#quick-start-)
  - [Requirements](#requirements-)
  - [Setup](#setup-️)
- [HOW TO RUN](#how-to-run-️)
  - [Baselines and Variants](#baselines-and-variants-)
  - [Multi-Objective Experiments](#multi-objective-experiments-)
  - [Output and Logs](#output-and-logs-)
  - [Play a Trained Policy](#play-a-trained-policy-)
  - [Media](#media-)
  - [Adding New Tasks](#adding-new-tasks-)

## Quick Start ⚡

### Requirements 🧰
- Isaac Sim compatible with **IsaacLab 2.1.0**
- Python 3.10
- CUDA-capable GPU

### Setup 🛠️
```sh
# clone the repo
git clone --recursive https://github.com/jjaykumarp-rai/GCR-PPO
git checkout jishnu/adapt-to-alpha

# create a conda env (isaacsim 4.5)
conda create -n alpha_throw4.5 python=3.10

conda activate alpha_throw4.5

# install isaaclab from current directory
./isaaclab.sh --install

# for creating a new task (optional)
./isaaclab.sh --new task_name # task_name will be a dir inside direct or manager_based in source/isaaclab_tasks/isaaclab_tasks/

# following are needed according to the original repository

# remove the IsaacLab-packaged RSL-RL wheel:
./isaaclab.sh -p -m pip uninstall rsl-rl-lib

# install the local RSL-RL fork: (for gcr-ppo I guess)
./isaaclab.sh -p -m pip install -e rsl_rl

# do for setting up the tasks
./isaaclab.sh -p -m pip install -e source/isaaclab_tasks/
```

Refer to the original [README](README.og.md) for details about GCR-PPO.

## HOW TO RUN ▶️

### Baselines and Variants 🏁
#### GCR-PPO (multi-head critic + priority-aware PCGrad):
```sh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<task-id> \
  --num_envs 4096 --seed 0 --headless \
  --video --video_interval 1000 \
  --use_critic_multi --use_pcgrad
```
#### Multi-head critic only (no conflict resolution):
```sh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<task-id> \
  --num_envs 4096 --seed 0 --headless \
  --video --video_interval 1000 \
  --use_critic_multi
```

### Multi-Objective Experiments 🎯
Supported example tasks:
- `Throwing-G1-General` (humanoid)
- `Isaac-Replicate-G1-Throw-Direct-v0` (alpha)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Replicate-G1-Throw-Direct-v0 \
  --num_envs 4096 --seed 0 --headless \
  --use_critic_multi --use_pcgrad \
```

See `scripts/reinforcement_learning/rsl_rl/train.py` for available flags ⚙️
(e.g., --energy, --gait, --armsp, etc.) and their ranges.

### Output and Logs 📊
Training artifacts (checkpoints, metrics, optional videos) are written to `logs/<task_name>` based on your CLI flags. Hydra configs and run traces live in `outputs/`.
To inspect training curves:
```sh
tensorboard --logdir logs/
```

### Play a Trained Policy 🎮
```sh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Replicate-G1-Throw-Direct-v0 --num_envs 4 --checkpoint=<path/to/checkpoint.pt>
```

### Media 🎥
- Front-directed throw sample: `media/g1-front-dir-12K.mp4`
- Random-direction throw sample: `media/g1-rand-dir-12K.mp4`
- Random-direction with velocity tracking: `media/g1-rand-dir-12K-vel-track.mp4`

### Adding New Tasks 🆕
For adding new (a) Direct (single-class) tasks and (b) Manager (cfg-driven) tasks, see the original [README](README.og.md).
