<div align="center">

<img src="https://media.luminis-sim.com/media/challenge/posters/robodojo_logo.png"></img>

<h2 align="center">RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies</h2>

<h2 align="center"><a href="https://robodojo-benchmark.com/">Webpage</a> | <a href="https://robodojo-benchmark.com/doc/">Document</a> | <a href="https://arxiv.org/abs/2607.04434">Paper</a> | <a href="https://robodojo-benchmark.com/community">Community</a> | <a href="https://robodojo-benchmark.com/leaderboard">Leaderboard</a></h2>

</div>

## This fork: RoboDojo + MiVerse

This repository builds on RoboDojo and adds **MiVerse as a separate benchmark**.
The original RoboDojo tasks, configurations, and scoring semantics are preserved;
MiVerse adds 24 mobile manipulation tasks using the shared simulation stack:
eight base tasks and their `_l` (language) and `_s` (spatial) variants.

Both benchmarks use `bash scripts/evaluate.sh <command> --bench <NAME>`:

- **MiVerse:** pass `--bench MiVerse`; see the [MiVerse guide](scripts/MiVerse/README.md).
- **RoboDojo:** pass `--bench RoboDojo` (the default); see the [CLI guide](scripts/README.md) and [official documentation](https://robodojo-benchmark.com/doc/).

Task discovery, environment checks, single-task launch, and batch scheduling are
shared. Benchmark-specific task definitions and scoring remain separate.
`scripts/MiVerse/miverse.sh` remains a compatibility alias for MiVerse.

This fork uses the recorded [XPolicyLab submodule](XPolicyLab) for policy
integration; see its [README](XPolicyLab/README.md) for the adapters
included here. The RoboDojo overview, news, leaderboard, and citations below
refer to the original benchmark. For this fork, start with the
[quick start](#miverse-quick-start) and [shared CLI guide](scripts/README.md);
upstream policy counts and examples describe the broader upstream project.

https://private-user-images.githubusercontent.com/88101805/619409345-cc074c5d-4567-4418-8a29-1385aaba9d5b.mp4

## News

- **September 16–17, 2026**
  - Fixed a single-frame discrepancy in `get_obs` and updated both the code and Hugging Face data. Rerunning evaluations across multiple models showed that the experimental results remain essentially unchanged.
  - Together with RoboTwin, fixed the channel ordering of the RGB data byte stream to keep both projects consistent. Please use this update with XPolicyLab commit [`bb9a0b5`](https://github.com/XPolicyLab/XPolicyLab/commit/bb9a0b5f5136a74503b679af830bfd0a3a837d5c) or later.
  - Updated the simulation assets for the `swap_T` task to make the objects less prone to slipping.
- **July 6, 2026** — Released the [RoboDojo paper](https://arxiv.org/abs/2607.04434) and open-sourced the code.

## ✨ Highlights

<p align="center">
  <img src="https://media.luminis-sim.com/media/home/teaser.png" width="70%"></img>
</p>

<p align="center"><em>Overview of RoboDojo. RoboDojo unifies efficient simulation evaluation and reproducible real-world testing for generalist robot manipulation, covering 42 simulation tasks, 18 real-world tasks, heterogeneous parallel simulation, RoboDojo-RealEval, XPolicyLab, and a continuously updated leaderboard.</em></p>

> RoboDojo is **eval-only** in this release: it provides the simulator client, benchmark tasks, asset/config validation, and result artifacts. Policy integration and policy servers are owned by [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md).

- 🌐 **Unified sim-and-real benchmark** — 42 simulation tasks and 18 real-world tasks across 3 robot embodiments for generalist robot manipulation.
- 🧭 **Five capability dimensions** — Generalization, Memory, Precision, Long-Horizon, and Open, designed to probe different skills rather than simple object or layout reskins.
- 🧗 **Challenging by design** — intentionally hard, diverse, long-horizon tasks that expose failures hidden by simpler benchmarks.
- ⚡ **Heterogeneous parallel simulation** — runs different tasks, scenes, and processes concurrently on Isaac Sim for fast, scalable feedback.
- 🧱 **Physically grounded assets** — rigid, articulated, and deformable objects in a single configuration-driven scene.
- 🤖 **Integrate once, evaluate everywhere** — [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md) unifies 40+ policies behind one interface for both simulation and real-world runs.
- 📊 **Reproducible & leaderboard-ready** — seed-controlled layouts and one-command `summarize` aggregation into a leaderboard table.

## 📚 Documentation

For the added benchmark, see the [MiVerse guide](scripts/MiVerse/README.md).
The [RoboDojo documentation](https://robodojo-benchmark.com/doc/) remains the canonical reference for the original RoboDojo benchmark. Key sections:

| Section | Description |
| :-- | :-- |
| [Usage Overview](https://robodojo-benchmark.com/doc/usage/) | End-to-end walkthrough of the evaluation workflow. |
| [Installation & Downloading (Assets and Data)](https://robodojo-benchmark.com/doc/usage/install-and-download/) | Environment setup and downloading robot/object/layout assets/training data. |
| [Quick Evaluation](https://robodojo-benchmark.com/doc/usage/quick-evaluation/) | Quickly dispatch XPolicyLab to run a policy for testing. |
| [XPolicyLab](https://robodojo-benchmark.com/doc/usage/xpolicylab/) | Integrates a large collection of policies and defines how to integrate new ones. |
| [Simulation Tasks Details](https://robodojo-benchmark.com/doc/sim-tasks/) | The 42 Isaac Sim tasks across five capability dimensions. |
| [Real Robot Tasks Details](https://robodojo-benchmark.com/doc/real-tasks/) | The 18 real-world tasks on Piper X, Piper, and ARX X5. |
| [Configurations](https://robodojo-benchmark.com/doc/usage/configurations/) | Simulator, scene, robot, and camera configuration options. |
| [Common Issues](https://robodojo-benchmark.com/doc/common-issue/) | Troubleshooting for installation, assets, GPU memory, and evaluation. |

## MiVerse quick start

The simulator installer targets **Linux x86_64**, an NVIDIA GPU, and Python 3.11.
Run these commands from the repository root. Policy dependencies are installed
separately from the simulator environment.

```bash
git clone https://github.com/ZixuanLi2004/RoboDojo.git
cd RoboDojo
git submodule update --init --recursive
bash scripts/install.sh --install
conda activate RoboDojo
bash scripts/init_assets.sh --bench MiVerse
```

`init_assets.sh --bench MiVerse` downloads MiVerse objects, rooms, robots, and
saved layouts together with shared assets. To prepare both benchmarks, use the
default `bash scripts/init_assets.sh` (equivalent to `--bench all`); use
`--bench RoboDojo` for RoboDojo alone. Both use the same Hugging Face asset
source, configured through `HF_REPO_ID`, `HF_REPO_URL`, and `HF_REVISION`.
The selected revision must contain the requested benchmark's assets; missing
packages cause an error. These assets are not bundled with this checkout.
See the [asset options](scripts/README.md#download-assets) and the
[MiVerse asset layout](scripts/MiVerse/README.md#installation-and-assets).

Prepare the included weight-free demo policy, then check one task and preview
its launch:

```bash
conda create -n demo-policy python=3.11 -y
conda activate demo-policy
bash XPolicyLab/policy/demo_policy/install.sh
conda activate RoboDojo

bash scripts/evaluate.sh tasks --bench MiVerse --check
bash scripts/evaluate.sh doctor --bench MiVerse \
  --task place_kettle_on_tv_stand --seed 0 --eval-num 1 \
  --policy-dir XPolicyLab/policy/demo_policy --policy-env demo-policy
bash scripts/evaluate.sh eval --bench MiVerse \
  --policy-dir XPolicyLab/policy/demo_policy \
  --task place_kettle_on_tv_stand --ckpt demo --policy-env demo-policy \
  --action-type joint --seed 0 --eval-num 1 --dry-run
```

After the required assets and environments are ready, remove `--dry-run` for
an actual one-episode interface check. `demo_policy` holds the observed posture
and commands zero base velocity; it does not solve the task or load a checkpoint.
For learned policies, install the selected adapter's dependencies and supply a
compatible checkpoint. The [policy compatibility list](scripts/MiVerse/README.md#policy-and-action-compatibility)
specifies which adapters require `joint` or `ee`; the CLI default is `ee`.

Selecting `--bench MiVerse` sets the default environment profile to
`miverse` (`env_cfg/miverse.yml`), with `config_name: miverse` for layouts and
results. Results are written under
`eval_result/MiVerse/<task>/<policy>/miverse/`. Use the
[MiVerse guide](scripts/MiVerse/README.md) for multi-GPU evaluation, split
server/client commands, resume, and the full 24-task × 3-seed summary.
Each task requires 25 episodes per seed. Summary reports group the base tasks,
`_l` variants, and `_s` variants into clean, language, and spatial dimensions.
MiVerse supports every evaluation command; RoboDojo capability `dimensions`
do not apply to its tasks. For the original benchmark, follow the
[RoboDojo CLI guide](scripts/README.md).

### Checks without simulation

With Python 3.11+, PyYAML, NumPy, and Bash 4+ available:

```bash
bash scripts/evaluate.sh tasks --bench RoboDojo --check
bash scripts/evaluate.sh tasks --bench MiVerse --check
bash -n scripts/evaluate.sh scripts/eval_policy.sh
```

These checks validate task inventories and shell syntax. They do not start
Isaac Sim or load learned policy weights. GPU simulation, asset loading, Pink
IK execution, and real checkpoint inference require validation on a configured
Linux evaluation machine. A successful dry run or doctor check alone does not
establish that runtime validation.

## 🗂️ Repository Structure

```text
env/                   simulator backbone and managers
env_cfg/               simulator, scene, robot, and camera configs
task/RoboDojo/         original RoboDojo tasks and task YAML configs
task/MiVerse/          24 mobile manipulation tasks and configs (8 base + 16 variants)
src/eval_client/       shared simulator client
scripts/evaluate.sh    shared CLI; select RoboDojo or MiVerse with --bench
scripts/internal/      shared task discovery, validation, launch, and batching
scripts/RoboDojo/      RoboDojo scoring and download helpers
scripts/MiVerse/       MiVerse scoring, guide, and compatibility CLI alias
scripts/eval_policy.sh shared simulator client entry
XPolicyLab/            policy server and policy integrations
Assets/                downloaded robot, object, room, material, and layout assets
```

`evaluate.sh summarize --bench RoboDojo|MiVerse` selects the summary script in
the matching benchmark folder and writes `eval_result/<benchmark>/_summary.md`
by default. Each benchmark retains its own scoring and completeness rules.

## 🔌 Policy Integration

Policies live in the [XPolicyLab submodule](XPolicyLab/README.md), which owns policy structure, dependencies, checkpoint layout, and server behavior. The shared evaluation path uses these policy-side files:

```text
XPolicyLab/policy/<POLICY_NAME>/eval.sh
XPolicyLab/policy/<POLICY_NAME>/setup_eval_policy_server.sh
XPolicyLab/policy/<POLICY_NAME>/setup_eval_env_client.sh
XPolicyLab/policy/<POLICY_NAME>/deploy.py
XPolicyLab/policy/<POLICY_NAME>/deploy.yml
```

`evaluate.sh eval` requires `eval.sh` for validation but launches the two setup
scripts directly for either benchmark. The selected benchmark determines task
lookup, the default environment profile, and result paths. The client setup
calls `scripts/eval_policy.sh`; `deploy.py`
provides the environment-side adapter, and `deploy.yml` declares transport,
batching, and policy-specific runtime settings. A policy's `eval.sh` is also
available as a direct policy-side entry point.

## 🏆 Leaderboard

View live rankings on the [RoboDojo Leaderboard](https://robodojo-benchmark.com/leaderboard).

**Simulation.** The full evaluation stack is open source, so you can debug locally and iterate on scores. Official RoboDojo-endorsed listings are submitted through the cloud evaluation pipeline with anti-cheating verification.

**Real world.** Real-robot leaderboard entries are accepted through the same cloud evaluation process; see the public documentation for protocol, rules, and submission details.

## 📝 Citation

```bibtex
% RoboDojo
@article{chen2026robodojo,
  title={{RoboDojo}: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies},
  author={Chen, Tianxing and Chen, Yue and Li, Zixuan and Tang, Junyuan and Su, Kailun and Wan, Weijie and Chen, Baijun and Lu, Haoran and Yan, Haowen and Su, Honghao and others},
  journal={arXiv preprint arXiv:2607.04434},
  year={2026}
}

% XPolicyLab
@article{community2026xpolicylab,
  title={{XPolicyLab}: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment},
  author={Community, XPolicyLab and Chen, Tianxing and Chen, Yue and Nian, Tian and Cai, Zijian and Chen, Guangyu and Lin, Wenwei and Liang, Qiwei and Xiang, Peicheng and Su, Kailun and others},
  journal={arXiv preprint arXiv:2608.09892},
  year={2026}
}
```

## 🙏 Acknowledgements

RoboDojo builds on [Isaac Sim](https://developer.nvidia.com/isaac/sim), [IsaacLab](https://github.com/isaac-sim/IsaacLab), [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena), [RoboTwin 2.0](https://github.com/robotwin-Platform/robotwin), [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), and [MagicSim](https://arxiv.org/abs/2606.17511). We thank the authors and maintainers for their open-source contributions to the robotics community.

Contact [Tianxing Chen](https://tianxingchen.github.io/) or [Yue Chen](https://yuechen0614.github.io/) if you have questions or suggestions.

## 🏫 Affiliations

RoboDojo is operated by **AI MMLab Club**, a non-profit, vendor-neutral organization, and is jointly maintained and supported by a global consortium of academic institutional partners. To preserve the fairness, neutrality, and independence of the official evaluation, RoboDojo does not involve commercial companies in its governance, operation, funding, sponsorship, compute, hardware, or other forms of project support. For inquiries from academic or non-profit partners regarding project collaboration or resource support, please contact [RoboDojoCommittee@gmail.com](mailto:RoboDojoCommittee@gmail.com).

<img src="https://media.luminis-sim.com/media/home/partners/affiliations.png"></img>

## ⚖️ License

Released under the [MIT License](LICENSE).


<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-475569?style=flat-square&logo=python&logoColor=white&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Sim 5.1" src="https://img.shields.io/badge/Isaac_Sim-5.1-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Lab 2.3" src="https://img.shields.io/badge/Isaac_Lab-2.3-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-475569?style=flat-square&labelColor=64748b" height="22"/>
</p>
