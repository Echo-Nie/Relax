# Encoder 分离部署 + Encoder/Rollout 重叠预取：实现方案

> 日期：2026-09-03
> 目标脚本：`examples/on_policy_distillation/mopd/run-mopd-qwen35-9b-8xgpu-colocate.sh`
> sglang 源：`/sgl-workspace/sglang` @ **v0.5.15.post1**（⚠️ `../sglang` 是 0.5.12 陈旧副本，别读）
> 硬件：8 × NVIDIA H20（97871 MiB/卡）
> 参照实现：[vllm-project/vime#370](https://github.com/vllm-project/vime/pull/370)（"[Rollout] Add vLLM encoder-prefill disaggregation"，2026-08-03 merged）+ [#381](https://github.com/vllm-project/vime/pull/381)（EPD 文档）
> 前置文档：[`mmlongbench-multiimage-mm-cache-and-preprocess-analysis.md`](./mmlongbench-multiimage-mm-cache-and-preprocess-analysis.md)（尤其 §5 / §10 / §11）、[`opd-student-teacher-imbalance-measurements.md`](./opd-student-teacher-imbalance-measurements.md)
>
> 本文两个需求：**(A)** 把 ViT 拆到独立 GPU 上部署；**(B)** 让 encoder 与 rollout 重叠——在跑第 N 步 rollout / 训练时，提前为第 N+1 步的数据算好 embedding 缓存。

______________________________________________________________________

> ## ⛔ 已实测，结论是不做 —— 见 [`encoder-disaggregation-measured-verdict.md`](./encoder-disaggregation-measured-verdict.md)
>
> (A) (B) 都实现完并在 GPU 上跑通了，缓存 93.5% 命中、ViT 前向 142 → 8，但 `rollout_time` 525.2s → 531.8s（+1.3%）。
>
> 判据是个**上界**而不是噪声论证：`mm_encode_time` mean 0.947s / `generate_time` mean 139.5s = **0.68%**。encoder 即使瞬时完成，`rollout_time` 也只能降不到 1%——rollout 是 generation-bound。§0.4 那道 G0 测量门的答案是**不过**。
>
> 本文以下内容作为设计与踩坑记录保留（其中 sglang 的四个 bug 修复、mooncake 端口冲突、缓存 spill/restore 机制**独立于 EPD 仍然有用**，见结论文档 §3）。

______________________________________________________________________

## 实现状态（2026-09-03）

**(A) 与 (B) 均已实现并通过 CPU 测试，但尚未在 GPU 上跑过端到端。** 决策已定：**冻结 ViT**，并在启动期强制校验。

| 改动 | 文件 |
| --- | --- |
| `worker_type: encoder` + 逐组 `env` + 两阶段启动 + 把 encoder 排除出 weight-sync/router/offload 视图 | `relax/distributed/ray/rollout.py` |
| **encoder 启动分发**（Relax 原先写死 `http_server.launch_server`，`--encoder-only` 会被静默忽略）、健康等待跳过 `/flush_cache`、encoder 的 server args 分支 | `relax/backends/sglang/sglang_engine.py` |
| 启动期校验：**必须 `--freeze-vision-model`**、单 encoder、必须有消费组、拒绝 OPD top-k 碰撞 | `relax/backends/sglang/arguments.py` |
| `/encode` 的 `cache_only` 运行时补丁（按 req_id 前缀短路 `send_with_url`） | `relax/backends/sglang/encode_cache_only_patch.py`（新） |
| 跨 step 预取驱动 | `relax/engine/rollout/encoder_prefetch.py`（新） |
| 预取挂钩（复用已有的 `prefetched_samples_ref`） | `relax/engine/rollout/sglang_rollout.py` |
| `RELAX_ENCODER_PREFETCH{,_CONCURRENCY,_TIMEOUT_S}` | `relax/utils/env.py` |
| EPD 拓扑与启动脚本（**baseline 脚本原样保留供 A/B**） | `examples/on_policy_distillation/mopd/sglang-epd.yaml`、`run-mopd-qwen35-9b-8xgpu-colocate-epd.sh`（均为新增） |
| 18 条 CPU 测试 | `tests/backends/sglang/test_encoder_disaggregation.py`（新） |

与下文方案的两点偏差：① 预取直接实现了方案 ③（`cache_only` 补丁），跳过了 PoC 的方案 ②，因为补丁只有一处短路、比"注册垃圾桶 sink"更小也更省；② 该补丁做在**运行时 monkeypatch**（`_launch_server_with_patches` 里，与 OPD patch 同款），不进 `docker/patch/`，因此不需要重建镜像。

### 🔴 GPU 实跑结论（2026-09-03，两次，均死在 step 0）：**当前 sglang 版本下走不通**

编排层全部验证通过：encoder 起在 GPU0 只占 6.3 GiB、EPD phase 1 收 URL 并注入、`load_format:auto` 生效、卡位图与设计一致、encoder 正确排除出 router/weight-sync/offload。teacher 4→3(TP1×3) 腾卡也成立。

但**数据面走不通**，已用最小复现（`scripts/debug/epd_minimal_repro.sh`，不含 Relax）坐实根因：

```
encode_receiver.py:1786 _process_waiting_requests
 → :823 _try_recv_mm_data
   → qwen_vl.py:612 get_mm_data
     → base_processor.py:303 build_input_ids
       → self._tokenizer.encode(prompt)
         ValueError: text input must be of type `str` ...
```

**sglang 的 EPD 接收侧只接受文本 prompt，会自己重新 tokenize；而 Relax 的 rollout 请求全是预展开的 `input_ids`。** 对照实验（同一容器、同一模型）：

| 请求形态 | 结果 |
| --- | --- |
| `/v1/chat/completions` + 图片（TP1 / TP2 / 1 encoder 带 2 个语言引擎 / 64 并发） | ✅ 全部 0.6–4.1s 正常 |
| `/generate` + `text` + 裸 base64（Relax 的传输形态） | ✅ 0.2s |
| **`/generate` + 预展开 `input_ids` + 图片（Relax 的真实形态）** | ❌ **scheduler 子进程崩溃 → SIGQUIT → 整个 server 死** |

⇒ **不是配置问题，是结构性不兼容**，而且正是 §5.2 早就预言过的那条（"encoder 不导出展开后的 `input_ids`，由 receiver 侧现算"）。`input_ids` 是 Relax 对齐 train/rollout token、算 loss mask 的地基，OPD pre-expanded patch 存在的全部理由就是保住它——不能为了 EPD 改回文本。

**唯一出路**：给 sglang 打补丁，让 `build_input_ids` / `get_mm_data` 在请求已自带 tokenize 好的 ids 时透传，而不是 `tokenizer.encode()` 重算。这是一笔独立的、需要吃透 receiver 侧 offset/grid 反推逻辑的工作，**在它落地之前 EPD 对 Relax 不可用**。

**三个 bug 修完后（16:09 第四次实跑）：数据面通了，卡在产能上。**

修复清单：① `--encoder-bootstrap-port` 逐引擎分配（原本写死 8997，两个语言引擎抢端口报 `[Errno 98]`）；② sglang `encode_receiver` 的 `input_text is None` 回落 + `build_input_ids` 的已展开检测（见上）；③ `SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL=0` —— **必须设在 language_only 消费端**，探测和驱逐都发生在它的 tokenizer manager 里，设在 encoder 组上完全无效（满载时 2s 探针连超 3 次就把 encoder 摘掉，之后静默回落本地 ViT，指标上看不出来）。

修完之后 EPD **真的跑起来了**：191 次 `/encode`、47 个真实多图 prefill batch、上述三类报错全部归零。但紧接着：

```
Timed out waiting for image embeddings for request ... [repeated 121x across cluster]
```

`SGLANG_ENCODER_RECV_TIMEOUT` 默认 180s。这不是 bug，是**产能**：rollout 一次性打出 256 请求 × 2.8–3.6 图 ≈ **750 张图的 ViT**，而 EPD 把原本摊在 **4 张卡**（2 引擎 TP2，且与生成天然重叠）上的 ViT 全部压到 **1 张 TP1 卡**上。

### ✅ 最终实测判决（2026-09-03 18:05 第五次实跑，4 步完整数据）

修掉**第四个** bug（见下）之后，EPD 第一次稳定推进：4 步全程 `Timed out / 503 / Global Restart` 全 0。

| | baseline（107 步） | EPD + 预取（step 0-3） | 差 |
| --- | --- | --- | --- |
| **`rollout_time`** | **6.7 m** | **8.7 m** | **+30%** ⬅ EPD 的纯信号 |
| `step_time` | 24.1 m | 21.3 m | −11.6%（**不是 EPD 的功劳**） |

逐步：`step_time` 1440.8 / 1233.4 / 958.3 / 1469.9 s；`rollout_time` 523.3 / 531.4 / 418.2 / 608.2 s。两个 run 都没开 `--rollout-shuffle`，数据顺序一致，可按同 step 比：baseline step3 rollout 434 s vs EPD 608 s，**+40%**。

**三条结论：**

1. **EPD 把它本该加速的相位拖慢了 30-40%。** encoder 往返 + embedding 传输的代价，超过了从语言引擎卸掉 ViT 省下的。
2. **`step_time` 的改善是 `--freeze-vision-model` 带来的，不是 EPD。** baseline 那 107 步跑的时候脚本里还没有这个 flag（2026-09-03 才加）。⇒ **只有 `rollout_time` 是干净的 EPD 信号。**
3. **跨 step 预取收益为零。** step0（冷 cache）523 s vs step1（热）531 s，**没有差别**。缓存修好后 ViT 只占 32/256 次请求，预取再把这一小块挪走自然什么也省不到——真正的成本在**每请求无条件跑的 HF preprocessing**（768 次）和 **embedding 传输**，这两样预取一个都动不了。预取本身每步耗 373-385 s，全压在训练相位跟 actor rank 0 抢 GPU0。

⚠️ n=4，本 workload 噪声带 ±27%（5 步累计 ±7%），+30% 刚过线，不算铁证；但方向明确。

**下一步（未做）**：唯一还对症的是 `--encoder-transfer-backend mooncake`（治传输：GPU-direct RDMA，省掉 TCP 那一跳和接收端两次拷贝；硬件已具备，7×mlx5 + nvidia_peermem + mooncake 已装）。**但它会关掉刚修好的缓存** —— `encode_with_mooncake`（`:2060-2147`）完全不碰 `mm_cache`（只存在于 `_encode`），换过去等于 ViT 工作量 ×8。所以必须**先把缓存移植进 mooncake 路径**才谈得上。

______________________________________________________________________

⇒ **这条就是 §0.4 那个 G0 问题的实测答案，而且指向相反**：ViT 的绝对量并没有小到可以塞进 1/4 的卡里。8 卡下"腾一张卡做 encoder"在产能上不成立。要继续只有两条路，且都跟 EPD 的初衷冲突：给 encoder 加卡（可总共只腾出 1 张），或者压低 rollout 并发（直接损失吞吐）。真正适合 EPD 的仍然是 §5 早就说过的那个场景——**多节点、多引擎，encoder 有独立预算**。

**仍未做**：§6 的 G0 测量门、G0.5（`--rollout-shuffle` 未单独 A/B）、以及全部收益验证。§0.4 的收益上限估算与 §8 的风险清单**依然成立**——而且现在连收益都还没机会测。

______________________________________________________________________

## 0. TL;DR

### 0.1 好消息：绝大部分已经现成

| 事实 | 位置 |
| --- | --- |
| sglang 0.5.15.post1 **自带完整 EPD**（encode-prefill-decode 分离） | `srt/disaggregation/encode_server.py`（4241 行）、`encode_receiver.py`、`server_args.py:2318-2350` |
| **Qwen3.5 在白名单内** —— `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` | `server_args.py:5585-5602`（不在白名单直接 raise） |
| `--encoder-only` 进程**只构造 ViT**，LM 从不实例化 ⇒ 1 张卡绰绰有余（ViT ≈ 0.85 GiB） | `models/qwen3_vl.py:1268-1272`（`if not config.encoder_only: self.model = ...`），`Qwen3_5ForConditionalGeneration` 继承自它（`models/qwen3_5.py:1633`） |
| **语言侧自动 dispatch，不需要客户端 prime** —— 这是与 vLLM/vime 的关键差异 | `managers/tokenizer_manager.py:612-613` 无条件调 `_handle_epd_disaggregation_encode_request`，`:2943-2972` 注释原文 *"dispatch to encoder by default"* |
| encoder TP 与 language TP **完全解耦**，无任何断言绑定；TP1 encoder + TP2 language 合法 | 唯一耦合是 `receive_count = language tp_size` 用于扇出（`encode_receiver.py:1436, 1772`） |
| Relax 侧**已有** `worker_type` / `engine_groups` 骨架（`regular` / `prefill` / `decode` / `placeholder`）+ 每组 `overrides` | `relax/distributed/ray/rollout.py:78-227`（校验 `:99-102`）、`sglang_engine.py:1782-1791`（PD 分支）、`:1811-1816`（overrides） |
| `--sglang-encoder-only` / `--sglang-language-only` / `--sglang-encoder-urls` **CLI 已存在**（每个 `ServerArgs` 字段自动镜像成 `--sglang-*`） | `relax/backends/sglang/arguments.py:214-263` |
| encoder 的 embedding cache 在 **host RAM**（`.cpu()` 之后才入 cache），不吃显存 | `encode_server.py:1699-1729`（`mm_embedding.cpu()` 在 `:1704`），`mem_cache/multimodal_cache.py:76`（纯 `OrderedDict` LRU） |

⇒ **需求 (A) 主要是 Relax 编排层的活，不需要碰 sglang 内核。** vime 的做法（`worker_type: encoder` 组 + 先起 encoder 收 URL + 注入 `language_only`/`encoder_urls`）可以几乎一比一搬过来，而且**比 vime 更简单**：vime 因为走 vLLM 的 `ECExampleConnector`，必须在 rollout 侧手写 `prime_encoder()` 逐图预热；sglang 的 language 端会自动 dispatch，这一步在需求 (A) 里可以完全省掉（需求 (B) 才会重新用到它）。

### 0.2 坏消息：三个硬约束

**B1 — encoder 进程没有权重更新 / 休眠 / cache 失效端点。**
`encode_server.py` 的全部路由只有 7 个：`/encode`、`/send`、`/scheduler_receive_url`、`/health`、`/health_generate`、`/start_profile`、`/stop_profile`（`:3793 / :4041 / :4081 / :4096 / :4097 / :4187 / :4219`，另加 `--enable-metrics` 时的 `/metrics`）。全文 grep `update_weights` / `release_memory_occupation` / `resume_memory_occupation` / `flush_cache` / `memory_saver` **零命中**，gRPC 版同理。它没有 `Scheduler` / `ModelRunner` / `TpModelWorker`，整套 sglang 权重同步机制在这个进程里根本不存在。

- ⇒ **ViT 不能热更**：`MMEncoder` 在 `encode_server.py:315-319` 一次性 `get_model(...)`，此后永不改动。唯一换权重的办法是杀进程重启。
- ⇒ **encoder 不能 sleep**：colocate 下其它引擎靠 `release_memory_occupation` 让位给 actor（`sglang_engine.py:1066-1075`、`rollout.py:595-607`、`teacher_manager.py:238-258`），encoder 做不到，它的显存是进程级常驻。
- ⇒ **cache 不能失效**：encoder 的 `MultiModalStaticCache` 按图片内容 hash 索引，**没有 weight-version 维度**（`encode_server.py:1690-1729`），策略更新后它会心安理得地吐旧 ViT 算的 embedding。

**B2 — `--language-only` 不省显存，而且会造成"两个 ViT"。**
Qwen3-VL 系的 `self.visual = Qwen3VLMoeVisionModel(...)` 是**无条件构造**的（`models/qwen3_vl.py:1244-1251`），`--language-only` 并不跳过；权重加载处的 `if (encoder_only or language_only) and name not in params_dict: continue`（`:1508-1513`）只是"缺模块时别 KeyError"的容错，不是过滤器——vision 参数在 params_dict 里，照常加载。MiMoV2 把理由写死在注释里（`models/mimo_v2.py:1055-1057`）："*Always build vision/audio encoders so P can fall back to local encoding when the EPD encoder is unreachable.*"

⇒ 语言引擎仍然持有一份完整 ViT，并在 encoder 不可达 / 未 dispatch 时**静默回落到本地 ViT**（`tokenizer_manager.py:867` 的 `"Encoder embedding not available, falling back to local mm processing"`）。好处是有兜底；坏处是**同一策略版本下，走 encoder 和走 fallback 的请求可能用的是两份不同的 ViT**，而且你在日志里只看到一条 warning。

**B3 — colocate 的 GPU 预算等式是硬断言。**
`opd_utils.py:438-463`（单 teacher）与 `:310-317`（MOPD）都要求
`args.rollout_num_gpus + resource["teacher"][1] == actor_total_gpus`，否则 `ValueError`；
`teacher_manager.py:74-82` 还有一条共享 PG 容量断言。
—— 好在 §3 给出的排法**天然满足这条等式，一行断言都不用改**。

### 0.3 结论：`--freeze-vision-model` 是这条路的总闸

把 B1 + B2 合起来看，问题的形状是：**ViT 是被训练的**（`--freeze-*` 五个 flag 全默认 `False`，`relax/utils/arguments.py:439-473`；且 Relax 在 `model_provider.py:320-330` 会把 bridge 自带的 `freeze_vision_model=True` **覆盖回 False**），所以 encoder 里那份 ViT 每步都会过期，而它既不能热更也不能清 cache。

| | **冻结 ViT**（`--freeze-vision-model` 或 `--freeze-params-name-list vision_model`） | **不冻结** |
| --- | --- | --- |
| encoder 权重同步 | ✅ 不需要 | ❌ 需给 sglang 打补丁加 `/update_weights_from_*` + 加入 NCCL 组（见 §4.6） |
| encoder cache 失效 | ✅ 永久有效，可跨 step、跨 epoch 复用 | ❌ 需再打一个 `/flush_cache` 补丁，且每步清空 = 预取白做 |
| B2 的"两个 ViT" | ✅ 两份完全相同，dispatch/fallback 数值一致 | ❌ 静默分叉 |
| 跨 step 预取（需求 B）的正确性 | ✅ **数学上精确** | ⚠️ embedding 天生滞后一次 weight sync |
| 附带收益 | Qwen3.5 的 `conditional_branch_sync` 混合模态梯度对齐**本来就没生效**（`conditional_branch_sync.py:50-65` 只认 megatron.bridge 的 `Qwen3VLModel`，Qwen3.5 走的是 `Qwen35VLModelProvider`）⇒ 冻结 ViT 顺带消掉[混合模态 grad-sync 死锁](../../CLAUDE.md)那一类风险 | — |

⇒ **建议：先决定要不要冻结 ViT。这是科学问题不是工程问题，需要你拍板。** 冻结后整件事退化成纯编排工作；不冻结则需要两个 sglang 补丁，并且需求 (B) 的预取只能接受"视觉分支滞后一步"的近似。

### 0.4 诚实的收益预期 —— 必须先过 G0 测量门

本仓 §5 在 2026-08-26 已经评过一次 EPD，结论是 **"当前单机 8 卡下不值得上"**。那份结论的三条理由，今天两条仍然成立、一条被本方案绕开：

| §5 的理由 | 今天是否仍成立 |
| --- | --- |
| "GPU 预算 ❌ 没卡" | **被绕开** —— §3 用 teacher TP2×2 → TP1×3 腾出 1 张卡 |
| "EPD 去重的正是 prefix cache 已经覆盖的那部分" | **仍成立**，而且更强：teacher 侧的 group 亲和路由**已经实现**了（`_pick_replica`，`on_policy_distillation.py:76-109`，按 `group_index` 把一个 group 的 8 个 sample 钉在同一副本），所以 teacher 那 8× ViT 冗余也早就没了 |
| "传输量更糟：embedding 119-320 MiB vs base64 PNG 几 MB" | **仍成立，且可量化**，见下 |

**上限估算（悲观）**：`SGLANG_VLM_CACHE_SIZE_MB` 100→1024 单独拿到 `rollout_time` −35.4%（§10），那一笔消掉的是 ~7/8 的 ViT 重算；按此外推，**剩余 ViT 大约只占 rollout 的个位数百分比**——正好落在本 workload 的噪声带内（单步 ±27%、5 步累计 ±7%，§10）。

**下限成本（可算）**：`--image-max-token-num 8192` 下单图 embedding = 8192 × 4096 × 2B = **64 MiB**（实测均值 ~39 MiB）。每步 32 prompt × 2.8-3.6 图 ≈ **90-115 张唯一图**，即 **3.9-6.4 GiB/step** 的 embedding。默认 `zmq_to_scheduler` 下这份数据要**按语言引擎的每个 TP rank 各发一份**（`receive_count = language tp_size = 2`，`encode_receiver.py:1436`），且发之前先 `.cpu()`（`encode_server.py:1704`）⇒ 每步 **8-13 GiB** 的 GPU→CPU→TCP→CPU→GPU 往返，**单机也没有 shm/CUDA-IPC 快路**（`encode_receiver.py:1400-1408` 的 `cuda_ipc` 只作用于 server 内部的 tokenizer→scheduler 一跳，不作用于 encoder→language）。对照：base64 PNG 每图几 MB。

⇒ **G0 测量门（在写任何代码之前做）**：
- **G0a（上限）**：现配置下 ViT 前向占 rollout 墙钟多少？取法：encoder 侧 `encoder_metrics_collector.observe_model_forward`（`encode_server.py:1701-1710`）不可用于现状，改用非 EPD 路径的 profile —— 用 `--sglang-profile-*` 抓一个 rollout step，量 `visual`/ViT 模块的累计 GPU 时间占比。
- **G0b（下限）**：把 G0a 的数与"每步 8-13 GiB 额外搬运"对比。
- **判据**：`G0a > 3 × G0b_estimated` 才继续；否则 EPD 只对**需求 (B) 的重叠**有意义（把 ViT 挪到训练相位的空窗里去，即便量不大也是纯赚），而不对"分离"本身有意义。
- 这条判据来自 §11 的教训原文：**"只有落在关键路径上的浪费才值钱"** ——#3 砍掉 7/8 的 CPU 一无所获，#4 只改了个 cache 上限拿到全部收益。

### 0.5 那为什么还值得做

1. **需求 (B) 改变了性质。** §5 评的是"分离能不能去重 ViT"（答案：不能，prefix/feature cache 已经去过了）。而**重叠预取不是去重，是把残余 ViT 整体挪出关键路径**——挪到 colocate 训练相位那段 encoder 本来 100% 空闲的窗口里。哪怕残余 ViT 只有 5%，只要它被完整隐藏，就是净赚 5%（前提是 §0.4 的搬运成本不反超）。
2. **§5 自己指出的适用场景就是"跨引擎共享 embedding"**，本方案正是单机版的它：1 个 encoder 服务 2 个 student 语言引擎，天然消掉 group 被 `cache_aware` 打散到两个引擎时的双份 ViT（§7 #2 记录的 0.71 vs 理论 0.875 缺口）。
3. **能力储备**：多节点 / 更大 ViT / 视频 / 高分辨率场景下 ViT 占比会显著上升，届时这套编排是现成的。

______________________________________________________________________

## 1. 参照实现：vime PR #370 做了什么，怎么映射到 Relax

vime 是 slime 的另一个衍生分支（`vime/ray/rollout.py`、`vime/rollout/on_policy_distillation.py` 与 Relax 同源同名），但它只有 vLLM 后端。PR #370 全部改动只有 8 个文件、384 行。

| vime 的做法 | 位置 | Relax 对应物 |
| --- | --- | --- |
| YAML `server_groups` 新增 `worker_type: encoder` | `vime/ray/rollout.py` | `EngineGroupConfig.worker_type`，`relax/distributed/ray/rollout.py:79`，校验 `:99-102` |
| **先起 encoder 组并同步等待，收集 URL** | `start_rollout_servers` 的 "Phase 1" | `start_rollout_servers`，`rollout.py:4342-4416` |
| 把 URL 注入非 encoder 组：`language_only: true` + `encoder_urls: [...]` | `_make_group(..., overrides_extra)` | 每组 `overrides`，`sglang_engine.py:1811-1816` |
| encoder 引擎**不注册进 router** | — | `skip_router_registration=True`，已有先例 `teacher_manager.py:202-211` |
| encoder 组强制 `enable_prefix_caching=False`（无 LM KV group） | `vllm_engine.py` | sglang 侧不需要：encoder 进程压根没有 KV pool |
| encoder GPU **计入** `--rollout-num-gpus` | 文档 FAQ | `rollout.py:4424-4426` 的 `assert config.total_num_gpus == args.rollout_num_gpus` 天然如此 |
| rollout 侧 `prime_encoder()` 逐图预热 | `vllm_rollout.py` | **需求 (A) 不需要**（sglang 自动 dispatch）；**需求 (B) 要用**，见 §5 |
| 明说的缺口：*"weight sync into a split encoder/language-only deployment is not yet covered"* | #381 文档 Limitations | 就是本文 §0.3 的 B1；Relax 必须正面回答 |

**结论**：vime 把最难的一半（RL 场景下 encoder 的权重同步）明确留白了，CI 里用 `update_weights: false` 跑的。Relax 不能照抄这个留白 —— 我们的 ViT 默认在训练。

______________________________________________________________________

## 2. sglang 0.5.15.post1 EPD 能力清单

### 2.1 Server args（`srt/server_args.py:2318-2350`，section 注释 `# Encode prefill disaggregation`）

| Flag | 行 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--encoder-only` | 2320 | `False` | 起一个只有 ViT 的 encoder server |
| `--language-only` | 2324 | `False` | 语言侧；**不省显存**（§0.2 B2） |
| `--encoder-transfer-backend` | 2328 | `zmq_to_scheduler` | choices 见 `:276`：`zmq_to_scheduler` / `zmq_to_tokenizer` / `mooncake` |
| `--encoder-urls` | 2335 | `[]` | 静态 encoder 列表 |
| `--encoder-bootstrap-port` | 2339 | `8997` | `EncoderBootstrapServer` 端口（跑在 language 侧 TokenizerManager 进程内，`tokenizer_manager.py:503-525`） |
| `--encoder-register-urls` | 2343 | `[]` | **encoder 主动向 bootstrap 注册自己**（方向易记反，`encode_server.py:3508-3578`，30 次 × 5s 重试） |
| `--enable-adaptive-dispatch-to-encoder` | 2347 | `False` | 开了之后单图 / batch>1 的请求走本地，多图（≥ `SGLANG_ENCODER_DISPATCH_MIN_ITEMS`=2）才发 encoder |
| `--enable-prefix-mm-cache` | 2075 | `False` | **强制要求 `--encoder-only`**（`:5555-5558` 直接 raise）；缓存 post-ViT `EmbeddingResult` |
| `--enable-mm-global-cache` | 2097 | `False` | mooncake-store 支持的跨节点 embedding 缓存 |

关键 env（`srt/environ.py:936-950`）：`SGLANG_ENCODER_RECV_TIMEOUT=180`、`SGLANG_ENCODER_HTTP_TIMEOUT=1800`、`SGLANG_ENCODER_MAX_BATCH_SIZE=8`、`SGLANG_ENCODER_PREPROC_WORKERS=8`、`SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL=10` / `_TIMEOUT=2`、encoder 路径的 `SGLANG_VLM_CACHE_SIZE_MB` 默认 **4096**（`encode_server.py:330`，与非 encoder 路径的 100 不一致，§5.4 已记过这条勘误）。

### 2.2 请求流（与 vLLM 的关键差异）

```
client ──/generate(image_data=base64)──▶ language-only server
                                         └─ TokenizerManager.generate_request        tokenizer_manager.py:589
                                            └─ if language_only: _handle_epd_..._request()   :612-613
                                               └─ should_dispatch = True (默认)             :2943-2972
                                                  └─ mm_receiver.send_encode_request()
                                                     └─ 后台线程 POST {encoder}/encode      encode_receiver.py:1690-1752
        ┌──────────────────────────────────────────────────────────────────┘
        ▼
   encoder server ── _process_mm_items(HF processor) ─▶ 查/填 mm_cache ─▶ ViT ─▶ .cpu()
        │                encode_server.py:1666            :1690-1729        :1704
        └── ZMQ PUSH（pickle 元数据 + 裸 tensor 帧）──▶ 每个 language TP rank 的 PULL socket
                                                        encode_receiver.py:777-832
                                                        └─ scheduler waiting_list 里挂着，
                                                           收齐才放进 new_recv_reqs
                                                           request_receiver.py:95, 216-234
```

**没有 client priming 这一步。** 兜底：无 encoder 注册 / 未 dispatch → `mm_inputs is None` → 本地 ViT（`tokenizer_manager.py:854-892`）。

### 2.3 端点清单与缺失清单

| | encoder (`--encoder-only`) | language (`--language-only`) |
| --- | --- | --- |
| `/health`, `/health_generate` | ✅ 但是**真跑一次 dummy ViT 前向**（`encode_server.py:4096-4185`，1px PNG） | ✅ |
| `/encode`, `/send`, `/scheduler_receive_url` | ✅ | — |
| `/generate`, `/v1/chat/completions` | ❌ | ✅ |
| `/update_weights_from_distributed` / `_from_tensor` | ❌ | ✅ |
| `/release_memory_occupation` / `/resume_memory_occupation` | ❌ | ✅ |
| `/flush_cache` | ❌ | ✅ |

### 2.4 并行度与拓扑

- encoder TP ⟂ language TP，**无断言**。encoder 内部约束只有一条：DP 模式要求 `dp_size>1 且 tp_size==1`（`encode_server.py:3670-3676`），不支持 TP×DP。
- 一个 encoder 可服务多个 language server；一个 language server 可挂多个 encoder。
- ⚠️ **多 encoder 时按 mm item 随机打散**（`encode_receiver.py:1912-1957`，`random.shuffle(encode_idx)`），**会直接打坏 `--enable-prefix-mm-cache` 的命中率**，且无 load-aware、无失败重试。⇒ **本方案坚持只用 1 个 encoder。**
- ⚠️ bootstrap 健康检查：10s 一次、**2s 超时**、连续 3 次失败就把 encoder 从列表里摘掉（`encode_receiver.py:96-98, 176-222`）。而 encoder 的 `/health` 是真 ViT 前向，满载时 2s 很容易超 —— 这与[三层 health check 与 router 误杀](./mmlongbench-multiimage-sglang-scheduler-hang-postmortem.md)是同一类坑。**必须设 `SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL=0` 关掉**（我们用静态 `--encoder-urls`，不需要动态发现）。被误摘的后果是静默回落本地 ViT，指标上看不出来。

______________________________________________________________________

## 3. GPU 预算改造（直接回答"腾一张卡给 Encoder"）

### 3.1 现状

```
PG（8 bundle，来自 resource["actor"]=8）
bundle  0   1   2   3   4   5   6   7
       └── student rollout ──┘└──── teacher ────┘
        2 engine × TP2          2 replica × TP2
       [0, rollout_num_gpus=4)  [4, 8)
actor（Megatron TP4×DP2）时分复用全部 8 张，靠 offload/onload 让位
```

### 3.2 你的提法 vs 建议的提法

你说的是"让**其中一个** teacher 副本用一张卡"，即异构 TP（一个 TP2 + 一个 TP1）。这在当前代码里做不到——`gpus_per_replica` 是**全局单值**（`opd_utils.py:287-292`：`replicas_per_teacher = gpus_per_teacher // gpus_per_replica`，要求整除），支持异构要改 teacher_manager 的副本构造。

**建议改成：teacher 全部 TP1、开 3 副本。** 同样腾出 1 张卡，且：
- **零代码改动**，只改脚本：`TEACHER_GPUS=3`、`TEACHER_NUM_GPUS_PER_ENGINE=1` → `3 % 1 == 0` ✅
- 副本数 2→3，配合已有的 group 亲和路由（`_pick_replica`）反而**分散更均匀**
- 记忆条目已记："副本数对吞吐中性（compute-bound）"、"小模型 TP1 多副本才安全（前提是 `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1`，脚本已开）"

**⚠️ TP1 的 logits 峰值会翻倍。** teacher 的 logits 是 `[chunk, vocab/TP]` fp32，vocab=248320：
- TP2 + `CHUNK_SIZE=8192`：8192 × 124160 × 4B = **3.79 GiB**
- TP1 + `CHUNK_SIZE=8192`：8192 × 248320 × 4B = **7.58 GiB**

⇒ 转 TP1 时把 `SGLANG_LOGITS_PROCESSER_CHUNK_SIZE` 从 8192 降到 **4096**，保持同样的 3.79 GiB 峰值。
权重侧：9B bf16 ≈ 18.5 GB 全在一张卡上，`--teacher-sglang-mem-fraction-static 0.4` → 0.4 × 97.8 = 39.1 GB 覆盖权重+KV，KV 池仍有 ~20 GB（teacher 是 prefill-only，实测 token usage 0.00-0.02），充裕。

### 3.3 新排法：encoder 放进 rollout 预算内，**断言天然满足**

关键洞察：`--sglang-config` 的总量断言是 `config.total_num_gpus == args.rollout_num_gpus`（`rollout.py:4424-4426`），而 colocate 断言是 `rollout_num_gpus + teacher_gpus == actor_gpus`。**只要把 encoder 作为 rollout SglangConfig 里的一个 group，两条断言自动成立，一行都不用改。**

```
bundle  0        1   2   3   4        5    6    7
       └encoder┘ └── student LM ────┘ └── teacher ──┘
        TP1       2 engine × TP2       3 replica × TP1
       [0, rollout_num_gpus=5)         [5, 8)

rollout_num_gpus(5) + teacher(3) == actor(8)   ✅ opd_utils.py:438-463 / :310-317
config.total_num_gpus(1+4=5) == rollout_num_gpus(5)  ✅ rollout.py:4424-4426
teacher_manager 容量：5 + 0 + 1×3 = 8 <= 8      ✅ teacher_manager.py:74-82
```

> **注意 bundle 顺序**：`start_rollout_servers` 里 `gpu_offset` 是按 group 顺序累加的（`rollout.py:4358, 4389, 4399`），所以 encoder group 必须**排在第一个**才落在 bundle 0。

### 3.4 encoder 常驻的代价（B1 的账）

encoder 不能 sleep，所以训练相位它仍占着 bundle 0：

| 项 | 量 |
| --- | --- |
| ViT 权重（Qwen3.5-9B vision tower，bf16） | **~0.85 GiB**（依据：`--sglang-mm-enable-dp-encoder` A/B 时 per-rank 权重 8.85→9.28 GB，+0.43 GB = 未分片 ViT 的一半，§10） |
| CUDA context + NCCL/ZMQ 缓冲 | ~0.5-1 GiB |
| ViT 前向瞬时激活（`SGLANG_ENCODER_MAX_BATCH_SIZE=8`） | ~1-2 GiB |
| embedding cache | **0（在 host RAM）** |
| **合计常驻显存** | **~2.5-4 GiB** |

⇒ actor 在 bundle 0 上少 ~2.5-4 GiB。Megatron 各 TP rank 对称分配，等于**全局按最差卡收紧**。当前实测峰值 78.7 / 97.8 GB，余量 ~19 GB，够；但 `--max-tokens-per-gpu 10240` 附近要留意，**上线前必须跑一次满载 step 验证不 OOM**。

______________________________________________________________________

## 4. Relax 侧实现清单

按改动量从小到大。全部集中在 3 个文件。

### 4.1 `relax/distributed/ray/rollout.py` —— 增加 `encoder` worker type

```python
# :99-102 附近，EngineGroupConfig.__post_init__ 的 worker_type 白名单
- if self.worker_type not in ("regular", "prefill", "decode", "placeholder"):
+ if self.worker_type not in ("regular", "prefill", "decode", "encoder", "placeholder"):
```

配套在 `ModelConfig` 上加两个派生属性（对齐已有的 `has_pd_disaggregation`，`:139-140`）：

```python
@property
def has_encoder_disaggregation(self) -> bool:
    return any(g.worker_type == "encoder" for g in self.engine_groups)

@property
def encoder_groups(self):
    return [g for g in self.engine_groups if g.worker_type == "encoder"]
```

同时新增**逐组 env**（§5.4 论证了为什么必须有）：

```python
# EngineGroupConfig 新字段
env: dict[str, str] = field(default_factory=dict)
# 消费点：rollout.py:530 附近，紧跟已有的 env_vars.update(build_runai_streamer_env_for_load(...))
env_vars.update(self.group_env)
```

校验（新增，参考 vime 的 "reject incomplete or mixed EPD topologies"）：
- 有 `encoder` 组时，必须至少有一个 `regular` 或 `prefill` 组作为消费者；
- **建议限制 `len(encoder_groups) == 1` 且该组 `num_gpus == num_gpus_per_engine`（单 encoder）**，理由见 §2.4 的随机打散问题。多 encoder 留到确有需要再放开。

### 4.2 `start_rollout_servers` —— 两阶段启动（`rollout.py:4342-4416`）

照搬 vime 的 Phase 1 / Phase 2，但省掉 EC connector：

```python
encoder_urls: list[str] = []
if model_cfg.has_encoder_disaggregation:
    # Phase 1: encoder 组必须先起并同步等待 —— 它的 URL 要注入语言组的启动参数
    for group_cfg in model_cfg.encoder_groups:
        group = _make_group(group_cfg, router_ip=None, router_port=None)   # 不进 router
        handles, port_cursors = group.start_engines(port_cursors)
        if handles:
            ray.get(handles)                      # 同步：URL 必须先拿到
        encoder_urls.extend(u for u in ray.get([e.get_url.remote() for e in group.engines])
                            if u is not None)
        server_groups.append(group)
    logger.info("EPD phase 1: %d encoder endpoint(s) ready", len(encoder_urls))

# Phase 2: 其余组照旧（deferred init），语言组额外注入 overrides
for group_cfg in model_cfg.engine_groups:
    if group_cfg.worker_type == "encoder":
        continue
    extra = None
    if encoder_urls and group_cfg.worker_type in ("regular", "prefill"):
        extra = {"language_only": True, "encoder_urls": list(encoder_urls)}
    ...
```

⚠️ Phase 1 的同步 `ray.get` 会把 encoder 的冷加载串到启动关键路径上。encoder 只加载 ViT（0.85 GiB），比 LM 快得多，但仍要在启动超时里留出余量。

### 4.3 `relax/backends/sglang/sglang_engine.py::_compute_server_args` —— encoder 分支

在现有 PD 分支（`:1782-1791`）旁边加：

```python
elif worker_type == "encoder":
    kwargs["encoder_only"] = True
    kwargs["enable_prefix_mm_cache"] = True        # 依赖 encoder_only，见 server_args.py:5555
    kwargs["disaggregation_mode"] = "null"         # encoder_only 与 prefill/decode 互斥
    # ❗ 关键：脚本全局是 --sglang-load-format dummy，靠首次 weight sync 填权重；
    #    但 encoder 永远不接收 weight sync（§0.2 B1），dummy 会让它一辈子用随机 ViT。
    #    与 teacher 同款处理：opd_utils.py:144-146 也 setdefault("load_format", "auto")
    kwargs["load_format"] = "auto"
    kwargs["enable_memory_saver"] = False          # encoder 没有 memory saver 集成
```

> 注意通用 passthrough 在 `:1799-1807`：任何 `args.sglang_<field>` 都会被扫进来。`--sglang-load-format dummy` 正是从这里进来的，所以上面的显式赋值必须**在 passthrough 之后**（或用 per-group `overrides`，优先级最高，`:1811-1816`）。**这条不做，encoder 会静默输出随机 embedding，而且没有任何报错。**

### 4.4 不注册 router、不注册 DCS

- **router**：`register_to_router`（`:888-946`）对非 regular worker type 在旧 router / slime router 上直接报错。encoder 不服务 `/generate`，绝不能进 worker 注册表。走 `skip_router_registration=True`（`teacher_manager.py:202-211` 已有先例）。
- **DCS / 权重同步**：`skip_dcs_registration=True`。这一步是**强制**的 —— 见 §4.6 的推导。

### 4.5 offload / onload：encoder 换成 cache spill / restore

`EngineGroup.offload/onload` 原本无条件对所有 engine 发 `release_memory_occupation`。encoder 没有这个端点，会 404 / 抛错，所以第一版直接 `return []` 跳过。

**但"跳过"把账算漏了一半。** encoder 常驻的其实是两笔钱：

| 项            | 大小                     | 训练相位能不能释放                    |
| ------------ | ---------------------- | ---------------------------- |
| ViT 权重 + CUDA context | ~2.5-4 GiB             | 不能——释放了就没法预取，这正是 (B) 要的      |
| embedding cache | `SGLANG_VLM_CACHE_SIZE_MB`（本配置 8 GiB） | **能**——训练全程没人读它 |

而且第二笔只在 **mooncake** 上是显存：`_encode()`（zmq 路径）缓存前先 `.cpu()`，缓存在 host RAM；`encode_with_mooncake()` 为了 D2D RDMA 把张量留在卡上。这就是 [[三层同名两义]] 那条坑的另一面。

⇒ 给 sglang 加一对端点 `POST /spill_mm_cache` / `POST /restore_mm_cache`，接回现有生命周期：

```python
def offload(self):
    if self.worker_type == "encoder":
        return [e.spill_mm_cache.remote() for e in self.engines if e is not None]
    return [e.release_memory_occupation.remote() for e in self.engines if e is not None]
```

**必须是 spill 不是 clear。** clear 会把预取刚算完的东西全丢掉——预取的收益等于零，还倒赔一次 D2H。spill 语义是「训练前 D2H 到 pinned host，下次 rollout 前 H2D 回来」。

实现上三个不显然的地方：

1. **先 drain 再 spill。** 完成的请求会一直攥着 embedding（`_forward_results[rid]` 持张量 + mooncake MR 注册在上面）直到 `send_timeout`（180s）超时清理；预取请求**永远不会有 `/send`**，所以必然吃满这 180s。不 drain 的话，rollout 刚结束时整个 cache 都还被引用着，spill 释放不出任何东西。
1. **drain 不能掀正在跑的 forward。** `_run_forward` 还在跑时把 `_forward_results[rid]` pop 掉，它回写结果时会 KeyError，白算一次。所以是「有界等待（默认 10s，`SGLANG_ENCODER_SPILL_DRAIN_TIMEOUT`）→ 只清完成的 → 没等到的连同它的 cache entry 一起留在卡上，下一步再收」。
1. **spill 期间新落的 entry 直接进 host。** 否则一个跨相位的预取会把刚还给 trainer 的显存重新吃回去。

zmq 路径上两个端点都是显式 no-op——尤其 restore：它要是无脑把 host cache 传上 GPU，就正好做反了。

⚠️ 剩下的 ViT 常驻部分仍是 §3.4 那笔账，无解（sglang 的 `--encoder-only` 没有 memory-saver 集成）。

### 4.6 权重同步：为什么必须走 `skip_dcs_registration`

现状（`weight_update/update_weight_from_distributed.py:273-339`）：

```python
world_size = sum(engine_gpu_counts) + 1     # +1 = 训练侧 rank 0
```

组是一个**扁平的 `init_process_group` + 逐 bucket `dist.broadcast`**（`:359-376`），元数据 `names/dtypes/shapes` 对每个 engine **完全相同**，没有任何 per-engine 过滤，也没有 vision 前缀过滤的 hook（我在 `weight_update/**` 与 `checkpoint_service/**` 全文找过，`if name.startswith("visual")` 之类的 skip 条件**零命中**）。

⇒ 把 encoder 塞进这个组只有两种结果：
- **要么全量收**：它得接收整个 9B 的 state dict（包括它根本没有的 LM 参数），而且 `encode_server.py` **压根没有 `/update_weights_from_distributed` 端点**（§0.2 B1）——直接 404，整组 broadcast 死锁。
- **要么不进组**：`skip_dcs_registration=True`，encoder 收不到任何权重。

**⇒ 唯一可行的 Phase-1 形态是 `skip_dcs_registration=True` + 冻结 ViT + `load_format: auto`（从 HF checkpoint 直接加载那份永不变的 ViT）。**

若将来确实要热更 encoder 的 ViT（即不冻结），需要三件事，都不小：
1. 给 `encode_server.py` 打补丁加 `/init_weights_update_group` + `/update_weights_from_distributed`（Relax 已维护 `docker/patch/sglang/v0.5.15.post1.patch`，有落脚点）；
2. 在 Relax 侧建**第二个** NCCL 组（独立 `group_name`），并在 `update_weight_from_distributed.py:125/:139` 的枚举循环里加 vision 名字过滤 —— 今天没有这个 hook；
3. 再补一个 `/flush_cache`，每步清空 encoder 的 `MultiModalStaticCache`（否则 §0.2 B1 第三条：cache 按内容 hash、无 weight version，会一直吐旧 embedding）。而清空 cache 等于**预取全部作废**，需求 (B) 归零。

### 4.7 参数与校验（`relax/utils/arguments.py` — 需先征得同意再改）

按 CLAUDE.md 的 "Ask First"，`arguments.py` 的改动需要确认。建议最小面：
- 不新增 CLI flag：拓扑走已有的 `--sglang-config <yaml>`；
- 只在 `sglang_validate_args` 加一条：有 encoder 组时若 `args.sglang_load_format == "dummy"` 且 encoder group 的 `overrides` 未覆盖 `load_format`，直接报错（把 §4.3 那个静默坑变成启动即失败）。

### 4.8 ⚠️ 与 OPD pre-expanded patch 的碰撞

`RELAX_OPD_PREEXPANDED_PATCH=1` monkeypatch 了 `QwenVLImageProcessor.process_mm_data_async`（`opd_sglang_patch.py:98-208`），用来处理 `image_data=[{"format":"opd_preexpanded_raw", ...}]` 这种自带展开 `input_ids` 的请求——它存在的理由就是**避免 detokenize→retokenize 打乱 token 对齐**。

而 EPD 的 dispatch 路径**整个绕过 `process_mm_data_async`**（mm_inputs 由 encoder 给，`input_ids` 由 receiver 侧 `build_input_ids` 现算，`encode_receiver.py:823-829`）。逐条核对：

| 请求 | 走不走 pre-expanded 格式 | EPD 影响 |
| --- | --- | --- |
| student rollout `/generate`（普通 base64） | ❌ 不走（`opd_sglang_patch.py:119` 干净 fall-through 到 `_orig`） | ✅ **安全**，可以放心上 EPD |
| **teacher prefill**（`opd_preexpanded_raw`） | ✅ 走 | ❌ **teacher 引擎绝不能设 `language_only`**。本方案也没打算动 teacher，记为硬约束 |
| **student-at-teacher-topk prefill**（`_student_prefill`，打到 student router） | ✅ 走 | ❌ **会碰撞**：student 引擎一旦 `language_only`，这个请求会被 dispatch 到 encoder，而 encoder 不认识 `opd_preexpanded_raw` |

本 run 用 `--opd-token-selection student_sampled` 且没开 `--opd-log-prob-top-k` ⇒ `topk_worker is None` ⇒ `_student_prefill` 不会被调用，**当前配置安全**。但这是个定时炸弹：
⇒ **必须加一条启动期校验**：`has_encoder_disaggregation and (opd_log_prob_top_k or teacher_self_topk)` → 报错。

______________________________________________________________________

## 5. 需求 (B)：Encoder / Rollout 重叠预取

### 5.1 时间轴与空窗

colocate 单步（依据 `relax/components/rollout.py:403-517` + `relax/components/actor.py:192-236` + `relax/backends/megatron/actor.py:724-843`）：

```
        ├──────────── rollout 相位 ────────────┤├────── train 相位 ──────┤
时间 ──▶ student 生成(9m17s)  teacher 尾巴(~9m51s)  log_probs→adv→optim→weight sync
GPU0   [encoder ★工作]        [encoder ★空闲]        [encoder ★空闲 ← 预取窗口]
GPU1-4 [student LM 忙]        [student LM 0% 空转]   [offload → actor 训练]
GPU5-7 [teacher 忙]           [teacher 忙]           [offload → actor 训练]
```

> student 侧 50% 空转是已实测的（`opd-student-teacher-imbalance-measurements.md` §1：`nvidia-smi` 连续 50s 采样，GPU0-3 全 0%）。**encoder 的空闲窗口比这还长**：整个 teacher 尾巴 + 整个 train 相位。

⇒ **预取窗口 = teacher 尾巴 + train 相位，占单步的一大半，且 encoder 在其中 100% 空闲。**

### 5.2 数据从哪来：第 N+1 步的图片其实早就在内存里了

两层现成机制：

**(a) `PrefetchBuffer`（DataSource actor 里的后台线程）** —— `relax/utils/data/streaming_dataset.py:457-689`
`--prefetch-max-cached` 默认 **256** 个样本（≈ 8 步），线程池里已经把图片 **decode 成 PIL 并 smart-resize**（`multimodal/image_utils.py:254-320`），甚至已经**跑过一遍 HF image processor**（`check_sample_length` → `data_utils.py:485-490`，结果只取 `len(input_ids)` 然后丢掉）。

> **🔴 顺带发现的现网问题（与 encoder 无关，可独立修）**：这个预取线程只在 `set_index_order()` 被调用时才启动，而它只有两个触发点，其一是 `StreamingDataset.shuffle()`，只在 `if self.args.rollout_shuffle:` 下调用（`data_source.py:189-190`）。**MOPD 脚本没有 `--rollout-shuffle`**（默认 `False`，`arguments.py:809-813`）⇒ **epoch 0 的预取线程从未启动**，每个样本走 `streaming_dataset.py:575-581` 的**串行同步 fallback**。加一个 `--rollout-shuffle` 可能就是一笔免费收益，建议单独 A/B。

**(b) 跨步 sample 预取 `prefetched_samples_ref`** —— `relax/engine/rollout/sglang_rollout.py:1323-1330`
第 N 步 `generate_rollout` **结尾**已经把第 N+1 步的 `get_samples.remote(...)` 发出去了，ObjectRef 存在 `GenerateState` 单例上（跨 `reset()` 存活，`:177-181`），在第 N+1 步开头 `:865-879` 消费。**这个 ref 恰好在 train 相位期间 resolve。**

### 5.3 Hook 点与三个投递方案

**Hook 点：`sglang_rollout.py:1327-1330`。** 保留现有的 ref 创建，另起一个后台 task：`ray.get(ref)` → 取出每个 group 的图片 → 投给 encoder → 完成后把结果挂在 sample 上。第 N+1 步 `:869-876` 处 await 这个 task。

怎么把图片投给 encoder，三个选项：

| 方案 | 做法 | 评价 |
| --- | --- | --- |
| **① 经语言引擎预热**（vime 的 `prime_encoder` 原样） | 向 language server 发 `max_new_tokens=1` 的图片请求，靠自动 dispatch 顺带填 encoder cache | ❌ **不可行**：train 相位语言引擎已 offload；且会白占语言引擎 prefill（每图 8192 token） |
| **② 直连 `/encode` + 丢弃 sink** | 先 `POST /scheduler_receive_url`（`receive_count=1`，指向我们自建的 ZMQ PULL socket），再 `POST /encode`；embedding 推过来后直接丢弃 | ⚠️ 可行、**零 sglang 改动**，但每步白搬 3.9-6.4 GiB 到一个垃圾桶。可作为 PoC |
| **③ 给 `/encode` 加 `cache_only` 开关**（推荐） | patch `encode_server.py:3793` 的 handler：`cache_only=True` 时跑完 `_process_mm_items` + ViT + 填 `mm_cache` 就返回，跳过 `send_with_url` | ✅ 无额外搬运、语义最干净。改动很小（`send_with_url` 在 `:2269-2345`，早退即可），落在已有的 `docker/patch/sglang/v0.5.15.post1.patch` 里 |

**推荐路径**：PoC 用 ②验证收益 → 确认有收益后换 ③。
两者都**必须开 `--enable-prefix-mm-cache`**（它依赖 `--encoder-only`，`server_args.py:5555`），否则预取算完就丢，真请求到达时还得重算。

> **实现落点后来前移了。** 上面写的 hook 点是 `generate_rollout` **结尾**，也就是本步全部 rollout 结束之后。实际落地时改成了**本步样本一提交就发**（`generate_rollout_async` 里 `submit_generate_tasks(samples)` 之后立刻），预取因此与本步 rollout 重叠，而不是挤在 rollout 尾巴到 train 之间。
>
> 有个坑值得记下来：**不能复用 `prefetched_samples_ref`**。那个字段是提交循环的消费源（`while state.remaining_batch_size < submit_target` 会多轮迭代），提前写进去的话，同一步的下一轮迭代会把下一步的 batch 当成自己的吃掉。所以另开了 `state.next_samples_ref` 专供早发路径，`generate_rollout` 结尾只做「提升」而不是二次抓取。
>
> 早发有两个守卫：`partial_rollout` 或 `dynamic_sampling_filter` 任一开启就退回原来的晚发位置——现有顺序是 `add_samples(aborted)` → `get_samples(next)`，提前抓会让下一步跳过本步 abort 掉的样本。本配置两个都没开。

### 5.4 cache 容量核算

- 每步唯一图：32 prompt × 2.8-3.6 图 ≈ **90-115 张**
- 单图 embedding：`--image-max-token-num 8192` 上限 8192 × 4096 × 2B = **64 MiB**，实测均值 ~39 MiB
- 一步 ≈ **3.9-6.4 GiB**；要容纳"预取的 N+1 步 + 正在消费的 N 步" ⇒ **≥ 2 步 ≈ 8-13 GiB**
- ⇒ encoder 侧需要 `SGLANG_VLM_CACHE_SIZE_MB=16384`（encoder 路径默认 4096 不够，会 LRU 抖动，重演 §3 的老问题）
- **代价是 host RAM，不是显存**（`encode_server.py:1704` 先 `.cpu()` 再入 cache，`multimodal_cache.py:76` 是纯 `OrderedDict`）—— 这条要在实机上用 RSS 复核一次

> **🔴 这里有个必须先解决的冲突：`SGLANG_VLM_CACHE_SIZE_MB` 是同名全局变量，但两条路径的语义完全不同。**
> - **非 encoder 路径**（student 语言引擎、teacher 引擎）：cache 存的是 **GPU 张量**（ViT 输出未经 `.cpu()`），且挂在模块全局 `embedding_cache` 上（`srt/managers/mm_utils.py:358`，`init_mm_embedding_cache`），引擎 sleep 时无人释放 ⇒ colocate 下每张卡在 actor 训练期间被永久扣掉这么多显存。这条是**实测**的，见 mm-cache 分析 §6 #4："每张 rollout/teacher 卡在 actor 训练期间被扣掉 ~1 GiB"。脚本现值 **1024**（≈1 GiB/卡，已验证 0 OOM）。
> - **encoder 路径**：cache 存的是 **CPU 张量**，默认 4096（`encode_server.py:330`）。
>
> ⇒ **绝对不能全局设 16384** —— 那会让 4 张 student/teacher 卡各被扣掉 16 GiB 显存，必 OOM。
> 必须做成**逐组 env**：encoder 组 16384，其余保持 1024。
> 现状 `EngineGroupConfig` **没有** `env` 字段，engine 的 `env_vars` 在 `rollout.py:496-545` 按组构造（已经会读 `self.sglang_overrides` 里的 `model_path` / `load_format`，`:529-533`），所以加一个 `env: dict[str, str] = field(default_factory=dict)` 并在 `:530` 附近 `env_vars.update(self.group_env)` 即可 —— 这是 §4 之外**额外的一条必做改动**。

### 5.5 正确性闸：权重滞后

预取必然发生在第 N 步的 weight sync **之前**，所以预取出的 embedding 用的是第 N 步的 ViT，而第 N+1 步的 rollout 本应使用同步后的 ViT。

| 前提 | 结论 |
| --- | --- |
| **ViT 冻结** | ✅ 滞后为 **0**，预取数学上精确；cache 可跨 step、跨 epoch 长期复用；language 侧本地 fallback 与 encoder 数值完全一致 |
| ViT 训练中 | ⚠️ 视觉分支 off-policy **1 步**。脚本当前 `--max-staleness 0`，等于偷偷破坏了这个设定。且 §0.2 B2 的"两个 ViT"会让**同一批样本里 dispatch 成功的和 fallback 的用不同权重**——这比统一滞后更糟。若走这条路，**必须先把 fallback 关死**（让 dispatch 失败直接报错而非静默回落），把不确定性变成确定的 1 步滞后，再用 A/B 看 reward / `opd_kl` 曲线 |

### 5.6 与 agentic 路径的关系

`relax/agentic/pipeline/prepare.py:53-65` 的 `PrepareDomain` + `--agentic-prelaunch` **已经是一套"提前一步做准备"的池**，而且 `runtime.py:280-370` 里已经在 prepare 阶段跑了 image processor + base64 编码。非 agentic 的 sglang 路径没有。⇒ 长期看这两条应该收敛到同一个抽象；本方案先不动 agentic。

______________________________________________________________________

## 6. 分期与验收

| 阶段 | 内容 | 通过判据 |
| --- | --- | --- |
| **G0**（先做，不写代码） | profile 一个 rollout step，量 ViT 前向占墙钟比例；对照 §0.4 的 8-13 GiB/step 搬运成本 | `G0a > 3 × G0b` 才继续做"分离"；否则只做需求 (B) 的重叠部分 |
| **G0.5**（独立小修） | 加 `--rollout-shuffle`，验证 `PrefetchBuffer` 线程真的起来了（看 `_prefetch_misses`） | `rollout_time` A/B，5 步累计超出 ±7% 噪声带 |
| **决策点** | 是否 `--freeze-vision-model` | 需要你拍板；不冻结则 Phase 2/3 的工作量翻数倍（§4.6） |
| **P1** | §4.1-4.5 + 4.8 校验；teacher 转 TP1×3；encoder 单卡起来 | 5 步跑完 0 OOM；`prefix_cache_hit_rate` 不下降；token 级输出与 baseline **逐 token 一致**（冻结 ViT 前提下应当精确一致，这是最强的正确性检查） |
| **P2** | §5 的预取（先方案 ②） | encoder `/metrics` 的 cache hit 率 > 90%；`rollout_time` 5 步累计 A/B |
| **P3** | 方案 ③ 的 `cache_only` patch | 搬运量降到 ~0，复测 P2 指标 |
| **P4**（可选，仅当不冻结 ViT） | §4.6 的三件套：encoder 权重更新端点 + 第二 NCCL 组 + vision 名字过滤 + flush_cache | — |

**A/B 纪律**（照抄 §10 的判据，别重蹈覆辙）：
- **单步差值 <27% 不可解读**，5 步累计约 ±7%；
- 先确认 `rollout/image_count/mean` 与 `rollout/multimodal_token_count/mean` 跨臂逐位相同（streaming dataset 下应当相同）；
- `response_len/mean` 是活的混淆项（temperature 1、无固定 seed）；
- ⚠️ **run 跑着的时候绝对不要改脚本**：bash 惰性读文件，改动会移动字节偏移，python 退出后可能从错位处执行碎片。要准备下一组配置就用 env 门控。

______________________________________________________________________

## 7. 脚本改动预览

新增 `examples/on_policy_distillation/mopd/sglang-epd.yaml`：

```yaml
sglang:
  - name: actor
    engine_groups:
      # ❗ encoder 必须排第一：gpu_offset 按 group 顺序累加，它要落在 bundle 0
      - worker_type: encoder
        num_gpus: 1
        num_gpus_per_engine: 1
        overrides:
          # 全局 --sglang-load-format dummy 靠首次 weight sync 填权重，
          # 但 encoder 永不接收 weight sync ⇒ 必须自己从 HF 加载真权重
          load_format: auto
        env:            # ⚠️ 需新增 EngineGroupConfig.env 字段，见 §5.4
          # encoder 路径的 cache 在 host RAM，可以放大；
          # 同名变量在语言/teacher 引擎上是 GPU 张量，全局设会 OOM
          SGLANG_VLM_CACHE_SIZE_MB: "16384"
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

脚本 diff：

```diff
 ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
-TEACHER_GPUS="${TEACHER_GPUS:-4}"
-TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-2}"
+TEACHER_GPUS="${TEACHER_GPUS:-3}"          # 4 -> 3，腾一张给 encoder
+TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-1}"   # TP2×2 -> TP1×3
+ENCODER_GPUS="${ENCODER_GPUS:-1}"
+ROLLOUT_TOTAL_GPUS=$((ROLLOUT_GPUS + ENCODER_GPUS))   # = 5，供 --rollout-num-gpus

-export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=8192
+# teacher 转 TP1 后 logits 是 [chunk, vocab=248320] 全词表不分片，
+# 8192 会让峰值从 3.79 GiB 翻到 7.58 GiB ⇒ 减半保持同等峰值
+export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=4096
+# ⚠️ SGLANG_VLM_CACHE_SIZE_MB 保持 1024 不动 —— 它对语言/teacher 引擎是 GPU 张量。
+#    encoder 那份（host RAM，需要 16384）走 YAML 的 per-group env，见 §5.4
+# encoder 的 /health 是真 ViT 前向，bootstrap 探针只有 2s 超时、3 次就摘除，
+# 满载时必误报；我们用静态 --encoder-urls，不需要动态发现 ⇒ 关掉
+export SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL=0

 SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2   # 保留！YAML 逐组覆盖 TP，但这个值还被
                                      # http_utils.py:251 的并发公式和 weight-sync 的
                                      # engine_gpu_counts 默认值用到，删掉会退化成 1
+   --sglang-config "${SCRIPT_DIR}/sglang-epd.yaml"
    ...
 )

+PERF_ARGS 里视需要加（决策点，见 §0.3）：
+   --freeze-vision-model

-RESOURCE_JSON="{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}"
+RESOURCE_JSON="{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_TOTAL_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}"

-    --rollout-num-gpus "${ROLLOUT_GPUS}" \
+    --rollout-num-gpus "${ROLLOUT_TOTAL_GPUS}" \
```

> `--sglang-encoder-transfer-backend` / `--sglang-language-only` / `--sglang-encoder-urls` 这些 CLI **已经存在**（`ServerArgs` 字段自动镜像，`relax/backends/sglang/arguments.py:214-263`）。但 `language_only` / `encoder_urls` 是**逐组**的，必须走 YAML `overrides`，不能用全局 CLI（全局设会让 encoder 组也变成 language_only）。

______________________________________________________________________

## 8. 风险与未验项

1. **收益可能在噪声带内。** §0.4 的外推给出"残余 ViT ≈ rollout 的个位数 %"，而本 workload 单步噪声 ±27%。**G0 不过就别做分离**。
2. **搬运成本可能反超收益。** 每步 8-13 GiB 的 GPU→CPU→TCP→CPU→GPU，单机也没有 shm 快路（`encode_receiver.py:1400-1408` 的 `cuda_ipc` 只管 server 内部那一跳）。可尝试 `--encoder-transfer-backend zmq_to_tokenizer` 把扇出从 `tp_size=2` 份降到 1 份、并让 tokenizer→scheduler 那一跳走 CUDA IPC；代价是 mm 接收变成 TokenizerManager 里的 inline await。**未验证。**
3. **`load_format: dummy` 陷阱**（§4.3）：不覆盖就是随机 ViT，**无任何报错**。P1 必须有逐 token 一致性检查兜底。
4. **encoder `/health` = 真 ViT 前向 + bootstrap 2s 超时 3 次摘除**（§2.4），误摘后静默回落本地 ViT，指标看不出来。除了关掉探针，还应在 P1 加一个"每步统计 dispatch 成功率"的打点。
5. **OPD top-k 与 EPD 碰撞**（§4.8）：当前配置安全，但改 `--opd-token-selection` / 开 `--opd-log-prob-top-k` 就炸。必须加启动期校验。
6. **encoder 常驻显存挤占 actor**（§3.4，~2.5-4 GiB，按最差卡全局收紧）。当前余量 ~19 GB，够但要实测。
7. **ViT 是否真的在收梯度，仍是"读默认值 + 三条间接佐证"**（§5.1 / §7 #7 已记：没有直接打点验 `requires_grad` / grad norm）。**冻结决策之前应该先花 10 分钟打一次点确认**——如果实际上 ViT 早就没在有效更新，整个 §0.3 的决策树会简化很多。
8. **`--rollout-shuffle` 缺失导致 PrefetchBuffer 未启动**（§5.2）—— 这条是独立发现，与 encoder 无关，值得单独验证和修复。
9. **多 encoder 的随机打散会打坏 cache 命中**（§2.4）。本方案锁定单 encoder；将来扩容前需要先给 sglang 加内容亲和的分派。
10. **Phase 1 同步启动**把 encoder 冷加载串进关键路径（§4.2），多节点场景下需要重新评估启动超时。
