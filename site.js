/* ─────────────────────────────────────────────────────────────
   Mermaid init — dark theme matching the rest of the site
   ───────────────────────────────────────────────────────────── */
if (window.mermaid) {
  mermaid.initialize({
    startOnLoad: true,
    theme: 'base',
    themeVariables: {
      darkMode: true,
      background: '#181818',
      primaryColor: '#1e1e1e',
      primaryTextColor: '#e8e8e8',
      primaryBorderColor: '#76b900',
      lineColor: '#76b900',
      secondaryColor: '#0d1a2e',
      tertiaryColor: '#141414',
      nodeBorder: '#76b900',
      clusterBkg: '#141414',
      clusterBorder: '#2a2a2a',
      edgeLabelBackground: '#141414',
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      fontSize: '13px',
    },
    flowchart: { curve: 'basis', padding: 12, useMaxWidth: true },
    securityLevel: 'loose',
  });
}

/* ─────────────────────────────────────────────────────────────
   Parallelism strategy picker
   Estimates per-GPU memory for training a dense transformer with
   Adam in BF16 and recommends the simplest strategy that fits.
   Assumptions (per param):
     - BF16 weights: 2 B
     - BF16 gradients: 2 B
     - FP32 Adam momentum + variance + master: 12 B
   Activations (with selective recompute, micro-batch 1, seq 4k):
   approximated as ~1.5 GB per billion params. Scales down with TP.
   This is deliberately an aid for sizing, not a replacement for
   NVIDIA's Calculator / running a real profile.
   ───────────────────────────────────────────────────────────── */
function recommendParallelism({ params_B, gpu_mem_GB, n_gpus }) {
  const safety = 0.85; // leave 15% headroom for CUDA context, NCCL, kernels
  const usable = gpu_mem_GB * safety;

  const weightsG    = 2    * params_B;
  const gradsG      = 2    * params_B;
  const optimG      = 12   * params_B;
  const actG        = 1.5  * params_B; // per GPU, micro-batch 1, seq 4k, recompute on

  const variants = [];

  // 1) Single GPU
  const singleGPU = weightsG + gradsG + optimG + actG;
  if (n_gpus >= 1 && singleGPU <= usable) {
    variants.push({
      rank: 1,
      label: 'Single-GPU training',
      detail: `Fits on one ${gpu_mem_GB} GB GPU with ~${singleGPU.toFixed(0)} GB used (of ~${usable.toFixed(0)} GB usable). Consider DDP across ${n_gpus} GPUs for throughput.`,
      mem: singleGPU,
      code: 'torch.nn.parallel.DistributedDataParallel',
    });
  }

  // 2) ZeRO-3 / FSDP over all n_gpus
  if (n_gpus >= 2) {
    const zero3 = (weightsG + gradsG + optimG) / n_gpus + actG;
    if (zero3 <= usable) {
      variants.push({
        rank: 2,
        label: `ZeRO-3 / FSDP across ${n_gpus} GPUs`,
        detail: `~${zero3.toFixed(0)} GB per GPU. Simplest path when model is too big for one GPU but cluster is small-to-medium. Weights + grads + optimizer all sharded.`,
        mem: zero3,
        code: 'torch.distributed.fsdp.FullyShardedDataParallel',
      });
    }
  }

  // 3) Tensor Parallelism at various sizes
  for (const tp of [2, 4, 8]) {
    if (n_gpus < tp) continue;
    const perTP = (weightsG + gradsG + optimG + actG) / tp;
    if (perTP <= usable) {
      const dp = Math.floor(n_gpus / tp);
      variants.push({
        rank: 3,
        label: `Tensor Parallel TP=${tp}${dp > 1 ? ' × DP=' + dp : ''}`,
        detail: `~${perTP.toFixed(0)} GB per GPU. Keep TP within a single NVLink domain (≤8 on HGX). Use Megatron-LM or NeMo for proven implementation.`,
        mem: perTP,
        code: 'megatron.core.tensor_parallel',
      });
      break;
    }
  }

  // 3b) Hybrid ZeRO-3 + TP=8 (FSDP "HSDP" / Megatron TP+DP with ZeRO)
  if (n_gpus >= 16) {
    for (const tp of [8, 4]) {
      if (n_gpus < tp * 2) continue;
      const dp = Math.floor(n_gpus / tp);
      const per = (weightsG + gradsG + optimG) / (tp * dp) + actG / tp;
      if (per <= usable) {
        variants.push({
          rank: 3,
          label: `Hybrid: TP=${tp} × ZeRO-3 across DP=${dp}`,
          detail: `~${per.toFixed(0)} GB per GPU. Sharding on 2 axes: TP within node, ZeRO-3 across nodes. The common pattern for 70 B+ pretraining.`,
          mem: per,
          code: 'torch.distributed.fsdp + megatron.core.tensor_parallel',
        });
        break;
      }
    }
  }

  // 4) 3D parallelism (TP×PP×DP) — fallback for huge models
  for (const tp of [8]) {
    for (const pp of [2, 4, 8, 16]) {
      if (n_gpus < tp * pp) continue;
      const dp = Math.floor(n_gpus / (tp * pp));
      if (dp < 1) continue;
      const shardFactor = tp * pp;
      const per = (weightsG + gradsG + optimG + actG) / shardFactor;
      if (per <= usable) {
        variants.push({
          rank: 4,
          label: `3D Parallel: TP=${tp}, PP=${pp}${dp > 1 ? ', DP=' + dp : ''}`,
          detail: `~${per.toFixed(0)} GB per GPU. PP stages via interleaved 1F1B schedule. Use Megatron-LM. Requires 8-GPU NVLink islands per TP group + IB NDR between PP stages.`,
          mem: per,
          code: 'megatron.core.pipeline_parallel',
        });
        return variants[0]; // earliest-fit wins
      }
    }
  }

  if (variants.length) return variants[0];

  return {
    rank: 99,
    label: 'Cluster too small',
    detail: `Even 3D parallelism across ${n_gpus} × ${gpu_mem_GB} GB does not fit a ${params_B} B model with Adam in BF16. You need either more GPUs, bigger memory (H200 141 GB / B200 192 GB), or aggressive offload (ZeRO-Infinity, CPU-offload on Grace-Hopper).`,
    mem: null,
    code: null,
  };
}

function mountParallelismPicker() {
  const root = document.getElementById('parallelismPicker');
  if (!root) return;

  const $params = root.querySelector('[data-field=params]');
  const $mem    = root.querySelector('[data-field=mem]');
  const $gpus   = root.querySelector('[data-field=gpus]');
  const $out    = root.querySelector('[data-result]');

  const update = () => {
    const params_B    = parseFloat($params.value) || 0;
    const gpu_mem_GB  = parseFloat($mem.value) || 0;
    const n_gpus      = parseInt($gpus.value) || 0;
    if (!params_B || !gpu_mem_GB || !n_gpus) {
      $out.innerHTML = '<span class="hint">Enter model size, GPU memory, and cluster size.</span>';
      return;
    }
    const r = recommendParallelism({ params_B, gpu_mem_GB, n_gpus });
    $out.innerHTML = `
      <strong>▸ ${r.label}</strong>
      <div class="hint">${r.detail}</div>
      ${r.code ? `<div class="hint" style="margin-top:.4rem">Key API: <code>${r.code}</code></div>` : ''}
    `;
  };

  [$params, $mem, $gpus].forEach(el => el.addEventListener('input', update));
  update();
}

/* ─────────────────────────────────────────────────────────────
   GPU picker — small decision helper (text-only, no Mermaid)
   ───────────────────────────────────────────────────────────── */
function recommendGPU({ phase, params_B, latencySensitive }) {
  if (phase === 'edge')
    return { label: 'RTX 6000 Ada / L4 / Jetson Orin', why: 'Dev workstations and edge inference — low power, PCIe form factor.' };

  if (phase === 'fine-tune')
    return params_B <= 13
      ? { label: 'Single H100 80 GB (QLoRA: A100 works)', why: 'LoRA/QLoRA for up to ~13 B fits on one 80 GB GPU. Add 2–8 GPUs for 70 B with FSDP.' }
      : { label: '8× H100 80 GB or H200 141 GB', why: 'Full-parameter fine-tune of 30–70 B needs FSDP or TP across a single HGX node.' };

  if (phase === 'inference')
    return params_B < 13
      ? { label: 'L40S 48 GB or L4 24 GB', why: 'Cost-optimal for <13 B serving; L40S runs quantized 13 B at high throughput.' }
      : params_B <= 70
      ? { label: 'H100 80 GB (×2–×8) or H200 141 GB (×1–×4)', why: 'H200 fits 70 B FP8 on fewer GPUs thanks to 141 GB HBM3e — saves NVLink-bound latency.' }
      : latencySensitive
      ? { label: 'GB200 NVL72 / B200 HGX (×8)', why: 'Blackwell FP4 + 2nd-gen Transformer Engine; NVL72 keeps 72-GPU TP-region in one NVLink domain.' }
      : { label: 'H200 (×8) with TP=8', why: 'For batch/offline serving of 70B+ dense, H200 is the cost sweet spot until B200 is widely available.' };

  // training
  return params_B < 7
    ? { label: 'Single 8× H100/H200 node', why: '<7 B fits on one HGX/DGX with FSDP. Megatron-LM or NeMo.' }
    : params_B <= 70
    ? { label: 'HGX H100/H200 cluster with InfiniBand NDR', why: '7B–70B training needs multi-node. 8–64 GPUs with NCCL over IB. TP=8 inside node, DP across nodes.' }
    : { label: 'DGX SuperPOD / GB200 NVL72', why: '70B+ frontier training. SuperPOD with SHARP in-network all-reduce, or GB200 NVL72 for coherent TP-72.' };
}

function mountGPUPicker() {
  const root = document.getElementById('gpuPicker');
  if (!root) return;
  const $phase  = root.querySelector('[data-field=phase]');
  const $params = root.querySelector('[data-field=params]');
  const $lat    = root.querySelector('[data-field=latency]');
  const $out    = root.querySelector('[data-result]');

  const update = () => {
    const phase = $phase.value;
    const params_B = parseFloat($params.value) || 0;
    const latencySensitive = $lat.value === 'yes';
    const r = recommendGPU({ phase, params_B, latencySensitive });
    $out.innerHTML = `<strong>▸ ${r.label}</strong><div class="hint">${r.why}</div>`;
  };
  [$phase, $params, $lat].forEach(el => el.addEventListener('input', update));
  update();
}

/* ─────────────────────────────────────────────────────────────
   TOC scroll-spy (right rail)
   ───────────────────────────────────────────────────────────── */
function mountTOC() {
  const rail = document.querySelector('.toc-rail');
  if (!rail) return;
  const links = rail.querySelectorAll('a[href^="#"]');
  const byId = new Map();
  links.forEach(a => byId.set(a.getAttribute('href').slice(1), a));

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      const a = byId.get(e.target.id);
      if (!a) return;
      if (e.isIntersecting && e.intersectionRatio >= 0.15) {
        rail.querySelectorAll('a.active').forEach(el => el.classList.remove('active'));
        a.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -60% 0px', threshold: [0, 0.15, 0.5, 1] });

  byId.forEach((_, id) => {
    const sec = document.getElementById(id);
    if (sec) observer.observe(sec);
  });
}

/* ─────────────────────────────────────────────────────────────
   Glossary tooltips — attach title to every abbr[data-glossary]
   ───────────────────────────────────────────────────────────── */
function mountGlossary() {
  const defs = new Map();
  document.querySelectorAll('#glossary .glossary-term').forEach(dl => {
    const term = dl.querySelector('dt')?.textContent?.trim();
    const defn = dl.querySelector('dd')?.textContent?.trim();
    if (term && defn) defs.set(term.toLowerCase(), defn);
  });
  document.querySelectorAll('abbr[data-glossary]').forEach(el => {
    const key = (el.dataset.glossary || el.textContent).toLowerCase();
    const defn = defs.get(key);
    if (defn) el.setAttribute('title', defn);
  });
}

/* ─────────────────────────────────────────────────────────────
   Factory infographic click → scroll to section
   ───────────────────────────────────────────────────────────── */
function mountFactoryClicks() {
  document.querySelectorAll('.factory-node[data-target]').forEach(n => {
    n.addEventListener('click', () => {
      const id = n.dataset.target;
      const sec = document.getElementById(id);
      if (sec) sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  mountParallelismPicker();
  mountGPUPicker();
  mountTOC();
  mountGlossary();
  mountFactoryClicks();
});
