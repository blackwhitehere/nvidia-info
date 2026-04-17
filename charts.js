/* Chart.js initialization for all benchmark charts on nvidia-info site.
   Each chart is wrapped in an IIFE-guard so a missing canvas doesn't throw. */

Chart.defaults.color = '#c8c8c8';
Chart.defaults.borderColor = '#2a2a2a';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

const PALETTE_GREEN = ['#76b900','#5a8e00','#3d6000','#96d900','#b8f040','#d4f980'];
const PALETTE_BLUE  = ['#00c8ff','#0096cc','#006499','#00e4ff','#66e8ff','#99f0ff'];
const PALETTE_MIX   = ['#76b900','#00c8ff','#ffd700','#ff6b35','#a855f7','#00e099'];

const axisBase = {
  grid: { color: '#1e1e1e' },
  ticks: { color: '#b5b5b5' }
};

function mountChart(id, cfg) {
  const el = document.getElementById(id);
  if (!el) return;
  new Chart(el, cfg);
}

/* ─────────────────────────────────────────────────────────────
   CHART 1 — Training throughput per GPU (LLaMA-2 70B)
   Source: NVIDIA MLPerf Training v4.0/v4.1 submissions, NVIDIA
   "H200 Tensor Core GPU" product brief. Numbers are tokens/sec
   per GPU on a standardized 8-GPU node with Megatron-LM.
   ───────────────────────────────────────────────────────────── */
mountChart('chartTraining', {
  type: 'bar',
  data: {
    labels: ['A100 SXM4\n(BF16)', 'H100 SXM5\n(BF16)', 'H100 SXM5\n(FP8)', 'H200 SXM\n(BF16)', 'H200 SXM\n(FP8)'],
    datasets: [{
      label: 'Tokens/sec per GPU (LLaMA-2 70B)',
      data: [1350, 2800, 4200, 3400, 5100],
      backgroundColor: PALETTE_GREEN,
      borderRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: true, ...axisBase, title: { display: true, text: 'tokens/sec per GPU', color: '#b5b5b5' } },
      x: { grid: { display: false }, ticks: { color: '#b5b5b5' } }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 2 — Inference throughput by engine (LLaMA-2 70B, TP=8)
   Source: vLLM v0.5 benchmarks, TensorRT-LLM release notes,
   SGLang RadixAttention paper (arXiv:2312.07104).
   ───────────────────────────────────────────────────────────── */
mountChart('chartInference', {
  type: 'bar',
  data: {
    labels: ['HF Transformers', 'TGI', 'vLLM', 'TensorRT-LLM\n(FP16)', 'TensorRT-LLM\n(FP8)', 'SGLang'],
    datasets: [{
      label: 'Output tokens/sec (LLaMA-2 70B, TP=8)',
      data: [320, 980, 2400, 3200, 4800, 2900],
      backgroundColor: PALETTE_BLUE,
      borderRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: true, ...axisBase },
      x: { grid: { display: false }, ticks: { color: '#b5b5b5', font: { size: 10 } } }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 3 — Memory footprint by quantization (Llama 70B weights)
   Source: bitsandbytes, AutoAWQ, GPTQ-for-LLaMa repos.
   ───────────────────────────────────────────────────────────── */
mountChart('chartMemory', {
  type: 'bar',
  data: {
    labels: ['FP32', 'BF16', 'FP8', 'INT8\n(GPTQ)', 'INT4\n(AWQ)', '2-bit\n(HQQ)'],
    datasets: [{
      label: 'GPU Memory Required (GB)',
      data: [280, 140, 70, 70, 35, 18],
      backgroundColor: ['#ff6b35','#ffd700','#76b900','#00c8ff','#a855f7','#00e099'],
      borderRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: true, ...axisBase, title: { display: true, text: 'weights only, GB', color: '#b5b5b5' } },
      x: { grid: { display: false }, ticks: { color: '#b5b5b5' } }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 4 — GPU generation peak BF16 TFLOPS + HBM bandwidth
   Source: NVIDIA datasheets for V100, A100, H100, H200, B200, GB200.
   Secondary axis: HBM bandwidth in TB/s.
   ───────────────────────────────────────────────────────────── */
mountChart('chartGpuGen', {
  type: 'line',
  data: {
    labels: ['V100\n(2017)', 'A100\n(2020)', 'H100\n(2022)', 'H200\n(2024)', 'B200\n(2025)', 'GB200\n(2025)'],
    datasets: [
      {
        label: 'BF16 TFLOPS (dense)',
        data: [125, 312, 989, 989, 2250, 2500],
        borderColor: '#76b900',
        backgroundColor: 'rgba(118,185,0,.12)',
        fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#76b900',
        yAxisID: 'y',
      },
      {
        label: 'HBM bandwidth (TB/s)',
        data: [0.9, 2.0, 3.35, 4.8, 8.0, 8.0],
        borderColor: '#00c8ff',
        backgroundColor: 'rgba(0,200,255,.08)',
        borderDash: [6, 4],
        tension: 0.3, pointRadius: 5, pointBackgroundColor: '#00c8ff',
        yAxisID: 'y1',
      }
    ]
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: '#c8c8c8', font: { size: 11 } } } },
    scales: {
      y: {
        beginAtZero: true, position: 'left', ...axisBase,
        title: { display: true, text: 'BF16 TFLOPS', color: '#76b900' },
      },
      y1: {
        beginAtZero: true, position: 'right',
        grid: { drawOnChartArea: false },
        ticks: { color: '#00c8ff' },
        title: { display: true, text: 'HBM TB/s', color: '#00c8ff' },
      },
      x: { grid: { display: false }, ticks: { color: '#b5b5b5' } }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 5 — Parallelism strategy trade-offs (radar)
   Subjective 0–10 scoring of common strategies along 6 axes.
   Higher is better EXCEPT AllReduce Volume and Impl. Complexity
   (lower = easier on system / developer).
   ───────────────────────────────────────────────────────────── */
mountChart('chartComm', {
  type: 'radar',
  data: {
    labels: [
      'AllReduce\nVolume (lower=better)',
      'Memory\nEfficiency',
      'Scaling\nEfficiency',
      'Impl.\nSimplicity',
      'Fault\nTolerance',
      'Checkpoint\nMaturity'
    ],
    datasets: [
      {
        label: 'Data Parallel (ZeRO-1)',
        data: [1, 2, 7, 9, 8, 9],
        borderColor: '#76b900', backgroundColor: 'rgba(118,185,0,.15)', borderWidth: 2,
      },
      {
        label: 'Tensor Parallel',
        data: [4, 7, 9, 3, 5, 7],
        borderColor: '#00c8ff', backgroundColor: 'rgba(0,200,255,.1)', borderWidth: 2,
      },
      {
        label: 'Pipeline Parallel',
        data: [8, 8, 8, 2, 6, 6],
        borderColor: '#ffd700', backgroundColor: 'rgba(255,215,0,.1)', borderWidth: 2,
      },
      {
        label: 'ZeRO-3 / FSDP',
        data: [3, 9, 8, 5, 7, 8],
        borderColor: '#ff6b35', backgroundColor: 'rgba(255,107,53,.1)', borderWidth: 2,
      },
    ]
  },
  options: {
    responsive: true,
    scales: {
      r: {
        min: 0, max: 10,
        grid: { color: '#2a2a2a' }, angleLines: { color: '#333' },
        ticks: { color: '#666', stepSize: 2, backdropColor: 'transparent' },
        pointLabels: { color: '#c8c8c8', font: { size: 10 } },
      }
    },
    plugins: {
      legend: { labels: { color: '#c8c8c8', font: { size: 11 } } }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 6 — Inference engine relative throughput (bars) + p50 TTFT (line)
   Throughput relative to HF Transformers baseline (1×).
   p50 TTFT = time to first token, shorter is better.
   Source: vLLM project benchmarks + TensorRT-LLM release notes.
   ───────────────────────────────────────────────────────────── */
mountChart('chartEngines', {
  type: 'bar',
  data: {
    labels: ['HuggingFace\nTransformers', 'TGI', 'LMDeploy', 'vLLM', 'SGLang', 'TensorRT-LLM\n(FP8)'],
    datasets: [
      {
        type: 'bar',
        label: 'Relative throughput (× HF)',
        data: [1, 3.1, 4.2, 7.5, 9.1, 15.0],
        backgroundColor: PALETTE_MIX,
        borderRadius: 6,
        yAxisID: 'y',
        order: 2,
      },
      {
        type: 'line',
        label: 'p50 TTFT (ms, lower=better)',
        data: [820, 310, 220, 180, 165, 95],
        borderColor: '#ffd700',
        backgroundColor: '#ffd700',
        tension: 0.25, pointRadius: 5,
        yAxisID: 'y1',
        order: 1,
      }
    ]
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: '#c8c8c8', font: { size: 11 } } } },
    scales: {
      y: { beginAtZero: true, position: 'left', ...axisBase, title: { display: true, text: '× HF baseline', color: '#b5b5b5' } },
      y1: {
        beginAtZero: true, position: 'right',
        grid: { drawOnChartArea: false },
        ticks: { color: '#ffd700' },
        title: { display: true, text: 'TTFT (ms)', color: '#ffd700' }
      },
      x: { grid: { display: false }, ticks: { color: '#c8c8c8', font: { size: 11 } } }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 7 — Quantization Pareto: accuracy drop vs speedup
   Source: AWQ (Lin et al. 2023), GPTQ (Frantar et al. 2022),
   SmoothQuant (Xiao et al. 2023), NVIDIA FP8 whitepaper, QLoRA paper.
   y = speedup over FP16 on H100, x = perplexity delta % on WikiText.
   ───────────────────────────────────────────────────────────── */
mountChart('chartQuantPareto', {
  type: 'scatter',
  data: {
    datasets: [
      { label: 'FP16',           data: [{ x: 0.0, y: 1.0,  r: 7 }], backgroundColor: '#ffd700' },
      { label: 'BF16',           data: [{ x: 0.02, y: 1.0, r: 7 }], backgroundColor: '#f0e68c' },
      { label: 'FP8 (E4M3)',     data: [{ x: 0.15, y: 1.8, r: 8 }], backgroundColor: '#76b900' },
      { label: 'INT8 SmoothQuant',data:[{ x: 0.20, y: 1.9, r: 8 }], backgroundColor: '#00c8ff' },
      { label: 'INT4 AWQ',       data: [{ x: 0.50, y: 3.2, r: 9 }], backgroundColor: '#a855f7' },
      { label: 'INT4 GPTQ',      data: [{ x: 0.70, y: 3.1, r: 8 }], backgroundColor: '#c084fc' },
      { label: 'NF4 (QLoRA)',    data: [{ x: 1.00, y: 2.5, r: 8 }], backgroundColor: '#ff6b35' },
      { label: '2-bit HQQ',      data: [{ x: 3.50, y: 3.6, r: 9 }], backgroundColor: '#00e099' },
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#c8c8c8', font: { size: 10 }, boxWidth: 8 } },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: +${ctx.parsed.x}% PPL, ${ctx.parsed.y}× speedup`
        }
      }
    },
    scales: {
      x: {
        title: { display: true, text: 'Perplexity increase (%) — lower is better', color: '#b5b5b5' },
        beginAtZero: true, ...axisBase
      },
      y: {
        title: { display: true, text: 'Speedup vs FP16 — higher is better', color: '#b5b5b5' },
        beginAtZero: true, ...axisBase
      }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 8 — Training memory breakdown by strategy (Llama 70B, BF16+Adam)
   Stacked bar. Values in GB *per GPU*, assuming 8-way cluster
   where applicable.  "Activations" uses selective recompute.
   Source: DeepSpeed ZeRO paper, Megatron-LM activation recompute
   paper, FSDP design doc.
   ───────────────────────────────────────────────────────────── */
mountChart('chartMemBreakdown', {
  type: 'bar',
  data: {
    labels: ['DP-only', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3 / FSDP', 'TP=8', '3D (TP=8,PP=4,DP=2)'],
    datasets: [
      { label: 'Params (BF16)',       data: [140, 140, 140, 18,  18, 18],  backgroundColor: '#76b900' },
      { label: 'Gradients (BF16)',    data: [140, 140, 18,  18,  18, 18],  backgroundColor: '#00c8ff' },
      { label: 'Optimizer states (Adam FP32)', data: [840, 105, 105, 13, 105, 26], backgroundColor: '#ffd700' },
      { label: 'Activations (recomp.)', data: [70, 70, 70, 70, 12, 12], backgroundColor: '#ff6b35' },
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#c8c8c8', font: { size: 10 } } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y} GB` } }
    },
    scales: {
      x: { stacked: true, grid: { display: false }, ticks: { color: '#b5b5b5', font: { size: 10 } } },
      y: {
        stacked: true, beginAtZero: true, ...axisBase,
        title: { display: true, text: 'GB per GPU', color: '#b5b5b5' }
      }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 9 — Inference latency vs throughput Pareto
   Each engine is a curve across batch sizes 1 → 256.
   Source: vLLM 0.5 benchmarks, TensorRT-LLM release notes,
   SGLang paper. Illustrative; verify on your workload.
   ───────────────────────────────────────────────────────────── */
mountChart('chartLatThru', {
  type: 'line',
  data: {
    datasets: [
      {
        label: 'HF Transformers',
        data: [{x:90,y:110},{x:180,y:135},{x:320,y:185},{x:520,y:340}],
        borderColor: '#888', backgroundColor: '#88888833', showLine: true, tension: 0.25,
      },
      {
        label: 'TGI',
        data: [{x:280,y:80},{x:620,y:130},{x:980,y:230},{x:1400,y:450}],
        borderColor: '#ffd700', backgroundColor: '#ffd70022', showLine: true, tension: 0.25,
      },
      {
        label: 'vLLM',
        data: [{x:480,y:75},{x:1200,y:120},{x:2400,y:210},{x:3600,y:420}],
        borderColor: '#00c8ff', backgroundColor: '#00c8ff22', showLine: true, tension: 0.25,
      },
      {
        label: 'SGLang',
        data: [{x:510,y:68},{x:1280,y:105},{x:2550,y:180},{x:3800,y:390}],
        borderColor: '#a855f7', backgroundColor: '#a855f722', showLine: true, tension: 0.25,
      },
      {
        label: 'TensorRT-LLM FP8',
        data: [{x:950,y:58},{x:2200,y:85},{x:4100,y:150},{x:5600,y:360}],
        borderColor: '#76b900', backgroundColor: '#76b90022', showLine: true, tension: 0.25,
      },
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#c8c8c8', font: { size: 10 } } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.x} tok/s, p99 TTFT ${ctx.parsed.y} ms` } }
    },
    scales: {
      x: {
        type: 'linear', beginAtZero: true, ...axisBase,
        title: { display: true, text: 'Throughput (tokens/sec, higher=better)', color: '#b5b5b5' }
      },
      y: {
        beginAtZero: true, ...axisBase,
        title: { display: true, text: 'p99 TTFT (ms, lower=better)', color: '#b5b5b5' }
      }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 10 — Cost per 1M tokens vs throughput (inference)
   x = $/1M output tokens (approx., GPU-hour / throughput).
   y = throughput tokens/sec.  Bubble size ~ GPU memory (GB).
   Source: GPU hourly rates (Lambda/Coreweave public pricing
   ~$2.50/hr A100, ~$3.50/hr H100, ~$5/hr H200). Numbers
   illustrative — validate on your workload.
   ───────────────────────────────────────────────────────────── */
mountChart('chartCost', {
  type: 'bubble',
  data: {
    datasets: [
      { label: 'A100 80GB + vLLM',         data: [{x: 0.85, y: 1200, r: 12}], backgroundColor: '#00c8ff' },
      { label: 'H100 80GB + vLLM',         data: [{x: 0.45, y: 2400, r: 12}], backgroundColor: '#76b900' },
      { label: 'H100 80GB + TRT-LLM FP8',  data: [{x: 0.22, y: 4800, r: 12}], backgroundColor: '#3d6000' },
      { label: 'H200 141GB + TRT-LLM FP8', data: [{x: 0.19, y: 6200, r: 16}], backgroundColor: '#b8f040' },
      { label: 'L40S 48GB + vLLM (13B)',   data: [{x: 0.35, y: 1800, r: 9}],  backgroundColor: '#a855f7' },
      { label: 'B200 + TRT-LLM FP8',       data: [{x: 0.11, y: 9800, r: 18}], backgroundColor: '#00e099' },
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#c8c8c8', font: { size: 10 } } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: $${ctx.parsed.x}/1M tok, ${ctx.parsed.y} tok/s` } }
    },
    scales: {
      x: {
        type: 'linear', beginAtZero: true, ...axisBase,
        title: { display: true, text: '$ per 1M output tokens (lower=cheaper)', color: '#b5b5b5' }
      },
      y: {
        beginAtZero: true, ...axisBase,
        title: { display: true, text: 'Throughput (tokens/sec)', color: '#b5b5b5' }
      }
    }
  }
});

/* ─────────────────────────────────────────────────────────────
   CHART 11 — All-reduce latency by fabric, by message size
   Grouped bars.  Approximate latencies for a 64-GPU all-reduce
   using NCCL 2.20 on H100 systems.  Smaller = better.
   Source: NVIDIA NCCL perf tests, Spectrum-X whitepaper.
   ───────────────────────────────────────────────────────────── */
mountChart('chartCollectives', {
  type: 'bar',
  data: {
    labels: ['1 MB', '100 MB', '1 GB'],
    datasets: [
      { label: 'NVLink 4 (intra-node, 900 GB/s)',     data: [0.008, 0.18, 1.4],   backgroundColor: '#76b900', borderRadius: 4 },
      { label: 'NVLink 5 / NVL72 (1.8 TB/s)',         data: [0.005, 0.10, 0.78],  backgroundColor: '#b8f040', borderRadius: 4 },
      { label: 'InfiniBand NDR 400G + SHARP',         data: [0.022, 0.55, 5.1],   backgroundColor: '#00c8ff', borderRadius: 4 },
      { label: 'Spectrum-X 400G Ethernet',            data: [0.028, 0.72, 6.4],   backgroundColor: '#a855f7', borderRadius: 4 },
      { label: 'Standard 100G Ethernet (RoCE)',       data: [0.070, 2.4,  22.0],  backgroundColor: '#ff6b35', borderRadius: 4 },
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#c8c8c8', font: { size: 10 } } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y} ms` } }
    },
    scales: {
      y: {
        type: 'logarithmic', ...axisBase,
        title: { display: true, text: 'All-reduce latency (ms, log scale)', color: '#b5b5b5' }
      },
      x: { grid: { display: false }, ticks: { color: '#b5b5b5' } }
    }
  }
});
