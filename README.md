# PrimingVLA

**Self-Supervised Action Priming for Parameter-Efficient Vision–Language–Action Adaptation**

PrimingVLA explores a simple adaptation principle: **learn action structure before introducing multimodal conditioning**. Built on π0.5 and LeRobot, it keeps the pretrained vision–language and action backbones frozen and learns compact prompt banks through a three-stage training schedule.

## Method

1. **Action priming:** learn action structure through self-supervised masked action reconstruction.
2. **Conditioning bridge:** introduce visual and language conditioning and jointly adapt the prompt banks.
3. **Flow adaptation:** continue task-conditioned flow-matching training.

CPC controls action-prompt updates relative to VLM-prompt updates during conditioned adaptation.

## Setup

Use Python 3.12+ and a CUDA-capable GPU for training. With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/LianYi233/PrimingVLA.git
cd PrimingVLA
uv sync --locked --extra training --extra pi --extra libero --extra evaluation
source .venv/bin/activate
```

See the [installation guide](docs/source/installation.mdx) for system dependencies. Prepare the LIBERO dataset, a π0.5 checkpoint, the PaliGemma tokenizer, and LIBERO simulation assets before running experiments.

## Training and evaluation

The launchers contain machine-specific defaults. Set `DATASET_ROOT`, `PRETRAINED_PATH`, `TOKENIZER_PATH`, `OUTPUT_ROOT`, and `LOG_ROOT` for your machine; check the cache paths in the scripts as well.

```bash
# Full method, seed 0: 750 priming + 250 bridge + 3,000 flow updates.
GPU_IDS=0 bash examples/training/train_pi05_revision_4000.sh full_reference 0

# Evaluate the trained policy on all four LIBERO suites.
# Set CKPT_ROOT to the OUTPUT_ROOT used above and BASE_OUTPUT for results.
GPU_ID=0 bash run_eval_pi05_revision_4000.sh full_reference all 0
```

The full method uses **4,000 cumulative updates**; its final checkpoint is named `003000` because the flow-stage counter starts after the 1,000 preparation updates. For evaluation, configure LIBERO's `config.yaml` with valid asset, BDDL, and initial-state paths.

See the [training and evaluation notes](examples/training/README_pi05_revision_4000.md) for path configuration and ablations. These scripts are available on this repository's `main` branch.

## Analysis tools

- [NTK analysis across training stages](examples/analysis/README_pi05_ntk_stages.md)
- [Attention visualization for LIBERO rollouts](examples/analysis/README_libero_attention.md)

## Acknowledgments and license

This repository builds on [LeRobot](https://github.com/huggingface/lerobot) and the [π0.5 / OpenPI](https://github.com/Physical-Intelligence/openpi) ecosystem, with evaluation on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). We thank their authors and contributors.

See [LICENSE](LICENSE) for the Apache-2.0 license and third-party notices.
