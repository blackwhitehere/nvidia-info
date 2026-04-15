# nvidia-info

Comprehensive interactive guide to NVIDIA's AI infrastructure stack for LLM training and large-scale API serving.

## What's Covered

- **GPU Hardware** — H100, H200, B200, GB200 NVL72 specs and comparison
- **System Software** — CUDA 12, cuDNN, cuBLAS, NCCL, Flash Attention
- **Distributed Training** — Megatron-LM, DeepSpeed, FSDP, 3D parallelism
- **Inference & Serving** — TensorRT-LLM, vLLM, SGLang, Triton, NVIDIA NIM
- **Data Center Networking** — NVLink 4/5, InfiniBand NDR, RoCE, SHARP
- **Orchestration** — Kubernetes, Slurm, DCGM, Ray
- **Developer Journey** — step-by-step toy snippets showing how the layers connect in practice
- **Benchmarks** — Performance charts comparing GPUs, inference engines, and parallelism strategies

## Live Site

Deployed via GitHub Pages from the `main` branch.

## Tech Stack

Single-page static site: HTML + CSS + Chart.js (vendored UMD bundle). No build step required.
