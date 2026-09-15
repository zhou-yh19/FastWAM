# FastWAM — LeRobot Training & Teleavatar 2.0 Deployment

A fork of [FastWAM](https://github.com/zhou-yh19/FastWAM) focused on one path end to end:
**train a world-action model on LeRobot-format robot data, then serve it on a Teleavatar 2.0
(TA2) dual-arm robot over ROS 2.**

Nothing here is specific to one task. You supply a LeRobot dataset; the configs, scripts and
deploy chain are parameterised by task name throughout.

> Looking for the original paper code (LIBERO / RoboTwin benchmarks)? Its configs still ship in
> `configs/{data,task}/`, but the benchmark documentation lives upstream at
> [zhou-yh19/FastWAM](https://github.com/zhou-yh19/FastWAM). This document covers the
> LeRobot + TA2 path instead.

| | |
|---|---|
| **Input** | LeRobot 2.x dataset (`meta/` + `data/` + `videos/`), 3 cameras, 72-d action/state |
| **Model** | Wan2.2-TI2V-5B video expert + 1.0B action expert (MoT), flow matching |
| **Output** | 16-d action chunks at 20 Hz → ROS 2 joint/gripper commands at 200 Hz |
| **Hardware** | Training: 8×A100/A800 80 GB, DeepSpeed ZeRO-1/2. Serving: 1 GPU ≥24 GB |

---

## Contents

- [How the pieces fit](#how-the-pieces-fit)
- [Install](#install)
- [Data: what the code expects](#data-what-the-code-expects)
- [Train](#train)
- [Deploy on TA2](#deploy-on-ta2)
- [Moving checkpoints between machines](#moving-checkpoints-between-machines)
- [Analysis tools](#analysis-tools)
- [Conventions worth knowing](#conventions-worth-knowing)
- [Further reading](#further-reading)
- [Citation](#citation)

---

## How the pieces fit

```
LeRobot dataset ──transcode──> *_lowres ──transcode──> *_mono
  meta/tasks.jsonl                                       │
        │                                                │
        └──precompute_text_embeds──> text_embeds_cache/   │
                                            │            │
                              configs/data/<task>.yaml ───┤
                              configs/task/<task>.yaml    │
                                            │            │
                                    launch_train.sh ──────┘
                                            │
                              runs/<task>/<RUN_ID>/
                                ├── checkpoints/weights/step_*.pt
                                ├── dataset_stats.json      ← must travel with the weights
                                └── config.yaml
                                            │
                        ┌───────────────────┴───────────────────┐
                        │                                       │
            start_local_serve_ws.sh                    make_task_map.py
            (GPU host, WebSocket)                      (taskmap.json)
                        │
            run_task_ws.sh  (robot host, ROS 2 + RTP video)
```

Two hosts are normal: a GPU box runs inference, the robot-side machine runs ROS 2 and
receives camera frames over RTP. They talk over WebSocket + msgpack.

---

## Install

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Then pre-generate the ActionDiT backbone once (it is interpolated from the Wan2.2 DiT, not
downloaded):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

`DIFFSYNTH_MODEL_BASE_PATH` must be set for every training and serving process — the Wan
component loader resolves all model paths relative to it.

The robot-side client has a different, much lighter dependency set and **must** run on the
system interpreter, because ROS 2 Humble's C extensions are built for it. See
[docs/GUIDE.md](./docs/GUIDE.md).

---

## Data: what the code expects

A LeRobot 2.x directory tree:

```
data/<your_dataset>/
├── meta/
│   ├── info.json          # fps, total_episodes, total_frames, feature shapes
│   ├── tasks.jsonl        # {"task_index": 0, "task": "<natural-language instruction>"}
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/episode_NNNNNN.parquet     # action + observation.state
└── videos/chunk-000/<camera>/episode_NNNNNN.mp4
```

Three cameras are expected by the TA2 configs: `head_camera`, `left_color`, `right_color`.
Each records a side-by-side stereo pair; only the **left eye** is used for training.

### 1. Transcode the video

Source video is typically 3840×1920 HEVC. Decoding one 33-frame window costs ~3.2 CPU-seconds,
which starves the GPU (~94 % of wall time spent waiting on data). Two stages fix that:

```bash
# stage 1: original SBS -> downscaled SBS
DATASETS="<your_dataset>" bash scripts/transcode.sh

# stage 2: SBS -> left eye only (what training consumes)
DATASETS="<your_dataset>" MODE=mono bash scripts/transcode.sh

# required: rewrite meta/info.json, or LeRobot still reports the source resolution
python scripts/patch_transcoded_meta.py --mode lowres data/<your_dataset>_lowres
python scripts/patch_transcoded_meta.py --mode mono   data/<your_dataset>_mono
```

`DATASETS` takes dataset **base names** (no `_lowres` / `_mono` suffix), relative to
`SRC_ROOT` (default `./data`). Omit it and the script discovers every dataset with the right
source suffix. Output keeps the LeRobot layout: `meta/` and `data/` are symlinked back to the
source, only `videos/` is rewritten. Re-running skips finished files.

### 2. Write two configs

```bash
cp configs/data/ta2_mono_template.yaml configs/data/<your_task>.yaml
cp configs/task/ta2_mono_template.yaml configs/task/<your_task>.yaml
```

In the **data** config set `dataset_dirs` (one or more `*_mono` dirs) and
`text_embedding_cache_dir`. In the **task** config point `override /data:` at your data config
name and fill in `wandb`. Both templates document every field inline.

Camera resolutions in `shape_meta.images` must match the transcoded video exactly — a
mismatch produces silently wrong results, not an error.

### 3. Precompute text embeddings

The umT5-XXL text encoder is ~11 GB. Training and serving both read cached embeddings instead
of loading it:

```bash
python scripts/precompute_text_embeds.py task=<your_task>
```

This reads instructions straight from each dataset's `meta/tasks.jsonl` and writes
`<sha256 of the formatted prompt>.t5_len128.wan22ti2v5b.pt` into `text_embedding_cache_dir`.
Because the filename is a hash of the exact prompt text, **any** wording difference — including
whitespace — produces a file the trainer will not find.

---

## Train

```bash
bash scripts/launch_train.sh <your_task>
```

That is the whole command. It starts a tmux session with two windows: `train`, and `prune` —
the latter is **not optional**, since each ZeRO state snapshot is ~80 GB and will fill the disk
without it.

| Variable | Default | Meaning |
|---|---|---|
| `ZERO` | `1` | DeepSpeed stage, 1 or 2 |
| `NPROC` | `8` | GPUs per node |
| `KEEP` | `2` | ZeRO state snapshots to retain |
| `SESSION` | `fastwam_<task>` | tmux session name |
| `CONDA_ENV` | `fastwam` | conda env to activate |
| `DRY_RUN` | — | print the command, start nothing |

Any extra argument is passed through to hydra, so resume and LR-anneal need no new config file:

```bash
# resume full state (weights + Adam momentum + LR schedule + dataloader cursor + RNG)
bash scripts/launch_train.sh <your_task> \
  resume=./runs/<your_task>/<RUN_ID>/checkpoints/state/step_<N> \
  additional_steps=<M>

# LR anneal: rebuild the schedule over the remaining steps
bash scripts/launch_train.sh <your_task> \
  resume=./runs/<your_task>/<RUN_ID>/checkpoints/state/step_<N> \
  resume_reinit_lr=true additional_steps=<M> \
  resume_warmup_frac=0.0 learning_rate=<lr at the resume point>

# ZeRO-2 on 4 GPUs
ZERO=2 NPROC=4 bash scripts/launch_train.sh <your_task>
```

Two things that bite:

- **`batch_size` must match the run you resume.** The sampler stores `batch_in_epoch`
  (a batch count, not a sample count), so changing `batch_size` silently resumes from the
  wrong position in the epoch. No error, no crash.
- **`max_steps` is derived** when left `null`:
  `steps/epoch = ceil(ceil(len(dataset) / (batch_size × n_gpu)) / grad_accum)`,
  and `len(dataset)` is the **frame count**. Sliding windows overlap heavily, so "one epoch"
  is not "every independent sample once".

Full detail, including the `trainer_state.json` arithmetic for changing batch size mid-run:
[docs/GUIDE.md](./docs/GUIDE.md).

---

## Deploy on TA2

Read [docs/GUIDE.md](./docs/GUIDE.md) before touching a real robot. The
short version:

### GPU host — start the server

```bash
TASK=<your_task> bash start_local_serve_ws.sh 10     # 10 = denoising steps
```

`RUN` and `STEP` default to the newest run and the highest step number; override either to pin
a specific checkpoint. Wait for `WebSocket server listening on ws://0.0.0.0:8000` before
starting the client. Health check: `curl http://<host>:8000/healthz`.

`--task` must name the config the checkpoint was **trained** with — hydra recomposes the data
and model dimensions from it, so a mismatch fails at load time (by design; it used to fail
much later and less clearly).

### Optional — a task library for switching instructions

To select among several instructions by short name at runtime, generate a task map:

```bash
python experiments/teleavatar_v2_deploy/server/make_task_map.py \
  --dataset-dir data/<your_dataset>_mono \
  --cache-dir data/text_embeds_cache/<your_task> \
  --out taskmap.json
```

Keys are derived from the dataset directory name (`_lerobot_20fps_mono` etc. stripped); a
dataset recording several instructions gets one key per `task_index`. Add or override entries
with `--instruction 'key=full instruction text'`. `taskmap.json` is a generated,
per-deployment artifact and is gitignored.

### Robot host — start the client

```bash
cd experiments/teleavatar_v2_deploy/client
./run_task_ws.sh <taskmap-key> --dry-run      # infer only, nothing moves
./run_task_ws.sh <taskmap-key>                # live
./run_task_ws.sh                              # omit the key: server's startup instruction
```

**Always do a `--dry-run` pass first.** The script also records a rosbag of camera frames,
policy action chunks and robot state for the whole session, which is what the analysis tools
below consume.

Pre-flight check: `bash experiments/teleavatar_v2_deploy/check_deployment.sh`.

---

## Moving checkpoints between machines

A checkpoint alone is not servable. These must travel together, from the **same run**:

| File | Why |
|---|---|
| `checkpoints/weights/step_*.pt` | the weights (~12 GB) |
| `dataset_stats.json` | action/state normalization — mixing runs silently drifts the action scale |
| `config.yaml` | the run's resolved config |
| `text_embeds_cache/` entries | the conditioning, byte-identical to training |
| `Wan2.2_VAE.safetensors` | ~1.4 GB, **not** inside the `.pt` (only `mot` + `proprio_encoder` are saved) |

The DiT shards (19 GB), ActionDiT payload (2 GB) and T5 encoder (11 GB) do **not** need to be
copied — serving sets `skip_dit_load_from_pretrain=True` and reads cached text embeddings.

Two scripts move exactly that set via KS3:

```bash
# on the training server (internal endpoint)
TASK=<task> RUN_ID=<run_id> STEP=step_<N>.pt \
  KS3_BUCKET=ks3://<bucket>/<prefix> \
  bash push_checkpoint_ks3.sh

# on the deploy machine (public endpoint) -- same three values
TASK=<task> RUN_ID=<run_id> STEP=step_<N>.pt \
  KS3_BUCKET=ks3://<bucket>/<prefix> PROJECT_ROOT=/path/to/FastWAM \
  bash pull_checkpoint.sh
```

The pull script stages incoming `configs/` in `configs_incoming_<RUN_ID>/` rather than
overwriting yours, and prints a diff for you to merge.

---

## Analysis tools

All under `experiments/teleavatar_v2_deploy/server/`. The ones that run the model need
`--task` plus a matching `--checkpoint` / `--dataset-stats` pair from the same run; the rest
only read a bag or a stream.

| Script | What it answers | Runs the model |
|---|---|---|
| `analyze_deploy_bag.py` | predicted vs commanded vs measured joint traces from a rosbag | no |
| `joint_error_report.py` | per-joint tracking error tables | no |
| `rtp_stream_export.py` | dump the RTP video stream to files | no |
| `bag_visualize_and_predict.py` | re-run the policy over a recorded bag, render a comparison video | yes |
| `offline_infer_from_export.py` | replay exported frames without a robot | yes |
| `viz_trainset_pred_compare.py` | sanity check: prediction vs ground truth on training samples | yes |

Latency:

```bash
bash bench_latency_ws.sh                      # single point, server's default instruction
bash bench_latency_ws.sh <taskmap-key> 12     # specific task and step count
bash bench_latency_ws.sh --sweep              # every served task × 8/10/12 steps
```

The sweep asks the server which tasks it serves (`available_tasks` in the connection
metadata), so it needs no task list of its own. Denoising steps are per-request — no restart
needed — but start the server with `--warmup-steps 8 10 12` before comparing, so every CUDA
Graph is warm.

---

## Conventions worth knowing

**Action / state layout.** Datasets store 72-d vectors;
`TeleavatarSelectTransform` slices them to the openpi convention:

```
raw 72-d:  positions[0:16]  = [L_arm(7), L_grip_pos, R_arm(7), R_grip_pos]
           velocities[16:32]
           efforts[32:48]   = [L_arm(7), L_grip_effort, R_arm(7), R_grip_effort]
           + optional EE / chassis / ...

model:     state  14-d = [L_arm(7), R_arm(7)]
           action 16-d = [L_arm(7), L_grip, R_arm(7), R_grip]
```

Gripper channels are **force**-controlled: the dataset stores effort in N·m, converted to a
platform trigger in [0, 1] before normalization and converted back on the robot side.

**Hydra composition.** `configs/train.yaml` is always the entry point; `data`, `model` and
`task` are all `null` there, and the task config fills them in. Every task config must begin
with `# @package _global_` — without it the keys nest under `task.` and training will not
start. `launch_train.sh` pre-checks this.

Override precedence, low to high: `train.yaml` → `data`/`model` config → task config → command
line.

**`dataset_stats.json` belongs to its run.** Normalization vector widths are validated at load
time, but two runs with the same dimensions and different statistics will load happily and
produce subtly wrong actions. Never mix them.

**Paths are repo-relative.** Scripts derive their own root; no absolute path is baked in.
Override `PROJECT_ROOT` when a deploy machine uses a different layout.

---

## Further reading

| Document | Contents |
|---|---|
| [docs/GUIDE.md](./docs/GUIDE.md) | config chain in depth, resume/anneal semantics, batch-size arithmetic, checkpoint transfer, two-host deployment, safety, rosbag analysis, troubleshooting |
| [upstream repo](https://github.com/zhou-yh19/FastWAM) | original FastWAM: LIBERO / RoboTwin training and evaluation |

---

## Citation

This fork adds LeRobot ingestion and TA2 deployment; the model and method are from the FastWAM
paper. If you use this work, please cite:

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```

The RoboTwin evaluation code is adapted from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).
