# Run RoboDojo and MiVerse in Docker

This image is the **RoboDojo simulation-evaluation side** — Isaac Sim 5.1,
IsaacLab, CuRobo, the RoboDojo Python stack, and the lightweight XPolicyLab client
used by `src/eval_client`. It includes policy client adapters but **no model environments or checkpoints**: your **policy server runs outside the container** and the sim
client talks to it over the default WebSocket transport (`protocol: ws` in
`deploy.yml`). The same image supports both benchmarks through
`scripts/evaluate.sh --bench RoboDojo|MiVerse`; the policy must support the
selected robot's observations and actions.

- Base image: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` (Ubuntu 22.04 + CUDA 12.8)
- Conda env inside the image: `RoboDojo` (Python 3.11)
- Project root inside the container: `/workspace/RoboDojo`

```mermaid
flowchart LR
  subgraph host [Host / another machine]
    policy["Policy server\n(your policy, or the bundled demo_policy)\nbind 0.0.0.0:PORT"]
  end
  subgraph container [RoboDojo container]
    client["scripts/evaluate.sh client --bench NAME\n-> Isaac Sim eval"]
  end
  client -->|"WebSocket ws://host:port"| policy
```

## TL;DR

On a machine with an NVIDIA GPU (driver ≥ 570):

```bash
sudo bash docker/install_docker_nvidia.sh   # one-time: Docker + NVIDIA Container Toolkit
newgrp docker                               # (or log out/in) so docker works without sudo
conda create -n demo-policy python=3.11 -y
conda run -n demo-policy python -m pip install -e ./XPolicyLab
ROBODOJO_CONDA_ENV=demo-policy bash docker/smoke_docker.sh run             # build the image, run a demo eval, assert PASS
```

The last command prints a per-phase `PASS/FAIL` table and writes a result file.
(Step 4 uses the bundled demo server in a separate policy environment; prepare
it as described below, or jump to [step 5](#5-evaluate-your-own-policy).)

Everything below explains each step and how to plug in your own policy.

## 1. Prerequisites

| Need | Details |
| --- | --- |
| NVIDIA GPU + driver | Driver `570.x` or newer (required by CUDA 12.8). Check with `nvidia-smi`. |
| Disk space | The image is very large (~200 GB with all Isaac Sim extensions cached) and the build cache adds more — keep **300 GB+** free. |
| OS | Linux (tested on Ubuntu 22.04). |
| Assets | A populated `Assets/` directory for the selected benchmark; it is mounted at run time, not baked into the image. |

Download assets on the host before running the container:

```bash
bash scripts/init_assets.sh --bench RoboDojo
# Or: --bench MiVerse for MiVerse; omit --bench to download both.
```

Both benchmarks use the same Hugging Face asset source. The selected
`HF_REVISION` must include the requested benchmark's packages; missing packages
cause an error. `HF_REPO_ID` and `HF_REPO_URL` can override the source.
MiVerse needs `Assets/Object/MiVerse/`, `Assets/Room/MiVerse/`,
`Assets/Robots/m92uw/`, and `Assets/Eval_Layout/MiVerse/miverse/`, together with
shared materials and backgrounds. Building the image does not download assets.
See the [asset options](../scripts/README.md#download-assets) and
[MiVerse guide](../scripts/MiVerse/README.md#installation-and-assets).

## 2. Install Docker + the NVIDIA Container Toolkit

If Docker and the toolkit already work with `--gpus all`, skip to step 3.

The bundled installer sets up Docker Engine, the NVIDIA Container Toolkit, the
`nvidia` runtime, and adds you to the `docker` group:

```bash
sudo bash docker/install_docker_nvidia.sh
newgrp docker      # or log out and back in
```

Verify the GPU is visible inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If not, see [Troubleshooting](#6-troubleshooting).

> **China networks:** the installer auto-detects a restricted network and switches
> to TUNA/USTC apt mirrors + Docker Hub registry mirrors. Force it with
> `USE_CN_MIRRORS=1` (or disable with `USE_CN_MIRRORS=0`). Details in
> [Appendix B](#appendix-b-china-networks).

## 3. Build the image

From the repository root:

```bash
docker build -t robodojo:cuda12.8 .
```

- The **first build takes ~1 hour** (it downloads Isaac Sim and compiles CuRobo's
  CUDA kernels). Later builds reuse the layer cache and are fast.
- Headless RTX rendering works out of the box — the required libraries and a
  single Vulkan ICD are baked in (see [Appendix A](#appendix-a-configuration-reference)).
- `smoke_docker.sh run` (step 4) builds the image for you, so you can skip this
  step if you go straight there.

> **China networks:** `smoke_docker.sh` passes the right mirror `--build-arg`s
> automatically. For a manual build, see [Appendix B](#appendix-b-china-networks).

## 4. Run the demo evaluation

The fastest way to confirm the whole loop works is the bundled end-to-end check:

```bash
ROBODOJO_CONDA_ENV=demo-policy bash docker/smoke_docker.sh run
```

It validates the selected benchmark/profile and host policy environment, then
runs: GPU-in-container → image build → start the bundled `demo_policy` server →
a container evaluation → assert the result file. A final table reports each
phase, and results land under `eval_result/`.

The default benchmark is RoboDojo (`stack_bowls`, profile `arx_x5`). To run the
same check for MiVerse after preparing its assets:

```bash
ROBODOJO_SMOKE_BENCH=MiVerse ROBODOJO_CONDA_ENV=demo-policy \
  bash docker/smoke_docker.sh run
```

MiVerse defaults to `place_bread_in_oven` and profile `miverse`, with
`config_name: miverse` for layouts and results. Set `ROBODOJO_SMOKE_TASK` or
`ROBODOJO_SMOKE_ENV_CFG` to override these defaults. The script validates the
benchmark, task, and profile in the host policy environment before building or
launching containers. Server and client use the same selection, and verification
requires this run's result at the exact benchmark/configuration path.
It also checks that `eval_time` covers the requested episode count, capped by
the task's native count, matching the simulator's behavior.

`demo_policy` holds the observed robot pose, so `success_rate: 0.0` is **expected** — the
point is that the pipeline renders, talks to the policy over WebSocket, steps the sim,
checks the reward, and writes a valid result:

```json
{ "success_rate": 0.0, "eval_time": 1, "score": 0.0, "details": { "0": { "success": false } } }
```

Watch progress live from a second terminal with `bash docker/smoke_docker.sh monitor`.

The demo server runs on the host. Select a separate policy environment explicitly:

```bash
conda create -n demo-policy python=3.11 -y
conda run -n demo-policy python -m pip install -e ./XPolicyLab
ROBODOJO_CONDA_ENV=demo-policy bash docker/smoke_docker.sh run
```

The existing `ROBODOJO_CONDA_ENV` option selects the host policy environment
(default: `RoboDojo`). The commands above select `demo-policy` explicitly to
keep policy-server dependencies separate from the simulator's pinned runtime.

## 5. Evaluate your own policy

The container is the simulator side; you run the **policy server** yourself and the
container connects over WebSocket (`protocol: ws`, same as `deploy.yml`).

**5.1 — Start your policy server** (outside the container). Bind it to `0.0.0.0`
(all interfaces), not `127.0.0.1`, so a container can reach it. See
[XPolicyLab](../XPolicyLab/README.md) for policy servers.

For example, after installing the Pi_05 policy environment and checkpoint:

```bash
bash scripts/evaluate.sh server --bench MiVerse \
  --policy-dir XPolicyLab/policy/Pi_05 --task place_bread_in_oven \
  --ckpt YOUR_CHECKPOINT --policy-env uv --action-type joint \
  --policy-port 9999 --bind-host 0.0.0.0
```

**5.2 — Run the eval** in the container, pointing `--policy-host`/`--policy-port`
at your server:

```bash
docker run --rm -it \
  --gpus all --network host --ipc host \
  -v "$PWD/Assets:/workspace/RoboDojo/Assets:ro" \
  -v "$PWD/Assets:$PWD/Assets:ro" \
  -v "$PWD/eval_result:/workspace/RoboDojo/eval_result" \
  -v "$HOME/.cache/warp:/root/.cache/warp" \
  -v "$HOME/.local/share/ov:/root/.local/share/ov" \
  -v "$HOME/.cache/ov:/root/.cache/ov" \
  robodojo:cuda12.8 \
  bash scripts/evaluate.sh client --bench MiVerse \
    --task place_bread_in_oven --env-cfg miverse \
    --policy-name Pi_05 \
    --policy-host 127.0.0.1 --policy-port 9999 \
    --ckpt YOUR_CHECKPOINT --action-type joint --eval-num 1
```

Required client flags: `--task`, `--policy-name` (a directory under
`XPolicyLab/policy/`), `--policy-host`, `--policy-port`. Common options: `--ckpt`,
`--eval-num N` (or `native`), `--action-type` (default `ee`), `--env-gpu` (default
`0`). Add `--dry-run` to print the resolved command without launching anything.
Keep `--bench`, `--env-cfg`, task, seed, and action type consistent with the
server. For RoboDojo, select `--bench RoboDojo` and a compatible task/profile
(the default profile is `arx_x5`).

**Mounts explained** (the last three are strongly recommended — they prevent slow
first starts):

| Mount | Why |
| --- | --- |
| `Assets → /workspace/RoboDojo/Assets:ro` | Robot/object/layout/material assets (not baked into the image). |
| `Assets → $PWD/Assets:ro` (same host path) | curobo configs bake **absolute host paths**; this second mount makes them resolve. See [Troubleshooting](#6-troubleshooting). |
| `eval_result → …/eval_result` | Persist results on the host. |
| `~/.cache/warp` | Reuse CuRobo's compiled kernels (else it recompiles for minutes on every start). |
| `~/.local/share/ov`, `~/.cache/ov` | Reuse Isaac Sim extensions/shaders (else Kit may stall pulling them). |

### Networking: where is your policy server?

| Policy server location | Run mode | `--policy-host` |
| --- | --- | --- |
| Same host as the container | `--network host` | `127.0.0.1` |
| Same host, bridge network | `--add-host=host.docker.internal:host-gateway` | `host.docker.internal` |
| Another machine | default networking | that machine's IP / hostname |

`localhost` inside a **bridge** container is the container itself, not the host —
this is the #1 reason "the simulator can't reach the policy server." The `client`
command checks that the policy server port is reachable before the WebSocket
handshake and prints guidance if the server is unreachable.

## 6. Troubleshooting

**GPU not visible (`nvidia-smi` fails in a container).** The NVIDIA Container
Toolkit isn't active. Re-run `sudo bash docker/install_docker_nvidia.sh`, then
`sudo systemctl restart docker`.

**"Cannot reach the policy server."** Make sure the server binds `0.0.0.0`, and
match the networking table in step 5. Quick check inside the container:
`nc -vz <host> <port>`.

**First eval seems to hang for minutes (high CPU, low GPU).** A fresh container
recompiles CuRobo's Warp kernels and/or pulls Isaac Sim extensions. Mount the host
caches (step 5) so both are reused. `smoke_docker.sh` does this automatically.

**`ValueError: /abs/host/.../X5A.urdf is not a file`.** curobo robot configs
(`Assets/Robots/**/curobo.yml`, generated by
`utils/update_embodiment_config_path.py`) bake in absolute host paths. Mount
`Assets/` a second time at its original host path (step 5), both read-only.
`smoke_docker.sh` does this automatically.

**Tiled cameras hang on the first rendered frame.** Already fixed in the image
(see [Appendix A](#appendix-a-configuration-reference)); make sure you run with
`--gpus all`.

## Appendix A: Configuration reference

**`smoke_docker.sh` env toggles**

| Env | Effect |
| --- | --- |
| `ROBODOJO_SKIP_BUILD=1` | Reuse the existing image (skip the ~1 h build) — fast iteration. |
| `ROBODOJO_CN_MIRRORS=1/0` | Force China build mirrors on/off (default `auto`). |
| `ROBODOJO_SMOKE_BENCH=RoboDojo\|MiVerse` | Select the benchmark and its default task/environment profile (default `RoboDojo`). |
| `ROBODOJO_SMOKE_ENV_CFG` | Override the environment profile; it must belong to the selected benchmark. |
| `ROBODOJO_SMOKE_TASK`, `ROBODOJO_SMOKE_PORT`, `ROBODOJO_SMOKE_POLICY`, … | Override task/port/policy/etc. (see the script header). |

**Volumes**

| Mount | Purpose |
| --- | --- |
| `-v "$PWD/Assets:/workspace/RoboDojo/Assets:ro"` | Assets (required at runtime, not baked). |
| `-v "$PWD/Assets:$PWD/Assets:ro"` | Resolve absolute host paths baked into curobo configs. |
| `-v "$PWD/eval_result:/workspace/RoboDojo/eval_result"` | Persist per-task eval artifacts on the host. |
| `-v "$HOME/.cache/warp:/root/.cache/warp"` (and `~/.local/share/ov`, `~/.cache/ov`, `~/.cache/nvidia`, `~/.nv`) | Isaac Sim / CuRobo caches — recommended for fast, offline-friendly starts. |
| `-e ROBODOJO_USD_ASSET_PREFIX=/path -v /host/usd:/path` | Point at a local Isaac Sim USD asset mirror (optional). |

**GPU selection.** `--gpus all` exposes every GPU; pin one with
`--gpus '"device=0"'`. Choose the Isaac Sim device with `--env-gpu ID`.

**Headless RTX (already handled).** Tasks render tiled cameras through Isaac Sim's
RTX path. The image bakes in `libxt6` (needed by Isaac Sim's MaterialX render
libs) and pins a **single Vulkan ICD** via `ENV VK_ICD_FILENAMES` /
`VK_DRIVER_FILES` (the toolkit also injects one at run time; two ICDs for the same
GPU wedge RTX init on the first frame). This requires `--gpus all` and
`NVIDIA_DRIVER_CAPABILITIES=all` (set by default).

**Build args.** `CUDA_IMAGE` (base image), plus the China mirrors `UBUNTU_MIRROR`,
`PIP_INDEX_URL`, `MINICONDA_URL`, `CONDA_CHANNEL_MIRROR` (see the `Dockerfile`
header and [Appendix B](#appendix-b-china-networks)).

**Validate the image without a policy server**

```bash
docker run --rm --gpus all --network host --ipc host \
  -v "$PWD/Assets:/workspace/RoboDojo/Assets:ro" \
  robodojo:cuda12.8 \
  bash scripts/evaluate.sh doctor --bench MiVerse --skip-policy
```

Use `--bench RoboDojo` to check the original benchmark instead.

## Appendix B: China networks

Everything auto-detects a restricted network; these are the manual knobs.

**Host install** (`install_docker_nvidia.sh`) — auto-uses TUNA (Docker CE) + USTC
(NVIDIA toolkit) apt mirrors and sets Docker Hub registry mirrors in
`/etc/docker/daemon.json`. Force with `USE_CN_MIRRORS=1`, disable with
`USE_CN_MIRRORS=0`.

**Image build** — `smoke_docker.sh` sets the mirror build-args when it detects (or
is told, via `ROBODOJO_CN_MIRRORS=1`) a China network. For a manual `docker build`:

```bash
docker build \
  --build-arg UBUNTU_MIRROR=mirrors.tuna.tsinghua.edu.cn \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg MINICONDA_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  --build-arg CONDA_CHANNEL_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda \
  -t robodojo:cuda12.8 .
```
