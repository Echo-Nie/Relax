# Encoder 分离（EPD）+ 跨步预取：实测结论

> 日期：2026-09-03
> 设计文档：[`encoder-disaggregation-and-overlap-prefetch-plan.md`](./encoder-disaggregation-and-overlap-prefetch-plan.md)
> 基线：`mopd-qwen35-9b-2026-09-01-report.md`（107 步，`rollout_time` 6.7m / `step_time` 24.1m，**未带** `--freeze-vision-model`）
> 本轮日志：`examples/on_policy_distillation/mopd/mopd-qwen35-9b-epd-2026-09-03-21:55:24.log`
> 硬件：8 × H20（97871 MiB/卡），colocate，Qwen3.5-9B，mmlongbench 多图

______________________________________________________________________

## 0. 结论

**EPD + 跨步预取在这个 workload 上不成立，建议不要继续投入。** 这是**第二次独立测量**得出同一结论，且这次没有"缓存没生效所以不算数"的退路。

判据不是步时噪声，是一个**上界**：

```
mm_encode_time  mean = 0.947s
generate_time   mean = 139.5s      ⇒  encoder 往返 = 单请求的 0.68%
```

**encoder 即使变成瞬时，`rollout_time` 也只能降不到 1%。** 这条不依赖跑多少步。rollout 是 **generation-bound**，不是 vision-bound。

实测印证：

| | step 0（冷缓存） | step 1（热缓存） |
| --- | --- | --- |
| cache 命中 | 224/366 (61%) | 116/124 (**93.5%**) |
| ViT 前向次数 | 142 | **8** |
| `mm_encode_time` mean | 0.785s | **0.947s** ↑ |
| `generate_time` mean | 120.7s | 139.5s |
| `image_processor_time` mean | 13.70s | 17.07s |
| `get_samples_time` total | 36.6s | **2.75s** ↓ |
| **`rollout_time`** | **525.2s** | **531.8s** |

砍掉 134 次 ViT 前向 → `rollout_time` **+1.3%**。

**最能说明问题的一格是 `mm_encode_time` 反而升了**（0.785 → 0.947s）。热缓存删掉了 94% 的 ViT 前向，encoder 往返却变慢了 ⇒ **这个往返的成本从来就不是 ViT**。它是 encoder 每请求无条件跑的 HF 预处理（在哈希查找**之前**执行，缓存跳不过）加传输。这解释了为什么"把 ViT 挪到独立卡上"这个动作从原理上就动不了关键路径。

### 与上一次测量的关系

| | 2026-09-03 早些时候 | 本轮 |
| --- | --- | --- |
| 传输 | zmq_to_scheduler | **mooncake**（GPUDirect RDMA） |
| 缓存 | 被 batch 路径绕过（0% 命中） | **93.5% 命中**，已移植进 `encode_with_mooncake` |
| 预取时机 | rollout 结束后才发 | 本步样本一提交就发 |
| 训练期显存 | encoder 缓存常驻 | **spill 到 host，还 7.4-8.2 GiB** |
| `rollout_time` 冷 → 热 | 523s → 531s | **525s → 532s** |

上一次可以辩解成"缓存没生效"。这次缓存、传输、预取时机、显存四项全部按设计工作，数字一模一样。

______________________________________________________________________

## 1. 今天做了什么

### 1.1 change ①：预取前移

预取原本在 `generate_rollout` **结尾**发（本步 rollout 全部结束之后），改成**本步样本一提交就发**（`generate_rollout_async` 里 `submit_generate_tasks(samples)` 之后）。

**一个会静默吃掉数据的坑**：不能复用 `prefetched_samples_ref`。那是提交循环的消费源（`while state.remaining_batch_size < submit_target` 会多轮迭代），提前写进去的话，同一步的下一轮迭代会把下一步的 batch 当成自己的吃掉。另开了 `state.next_samples_ref` 专供早发路径，`generate_rollout` 结尾只做「提升」而不是二次抓取。

两个守卫：`partial_rollout` 或 `dynamic_sampling_filter` 任一开启就退回晚发位置——现有顺序是 `add_samples(aborted)` → `get_samples(next)`，提前抓会让下一步跳过本步 abort 掉的样本。

**唯一兑现的收益**：`get_samples_time` 36.6s → 2.75s。但被 generation 的波动整个吞掉。

### 1.2 change ②：encoder 缓存 spill / restore

约束 1「encoder 不能 sleep」其实可以拆成两半，有一半是能解的：

| 项 | 大小 | 训练相位能否释放 |
| --- | --- | --- |
| ViT 权重 + CUDA context | ~2.5-4 GiB | 不能（释放了就没法预取） |
| embedding cache | 本配置 8 GiB | **能**（训练全程没人读） |

而且第二笔只在 **mooncake** 上是显存（`_encode` 缓存前先 `.cpu()`，`encode_with_mooncake` 为了 D2D RDMA 留在卡上）。

⇒ 给 sglang 加了 `POST /spill_mm_cache` + `/restore_mm_cache`，Relax 侧把 `EngineGroup.offload/onload` 对 encoder 组的 no-op 换成这两个调用。编排时序**一行没改**——本来就在调。

**必须 spill 不能 clear**：clear 等于把预取刚算完的全丢掉。

三个不显然的地方，实测都是真的：

1. **先 drain 再 spill。** 完成的请求会一直攥着 embedding（`_forward_results[rid]` 持张量 + mooncake MR 注册在上面）直到 `send_timeout`=180s；预取请求**永远不会有 `/send`** 所以必吃满。实测 `drained=117`——127 个 entry 里 117 个当时还被引用着，**不 drain 的话这次 spill 一个字节都释放不出来**。
1. **drain 不能掀正在跑的 forward。** pop 掉 `_forward_results[rid]` 会让 `_run_forward` 回写时 KeyError，白算一次。所以是「有界等 10s → 只清完成的 → 没等到的连同 cache entry 留在卡上下步再收」。实测三次 spill 全部 `still_running=0`，没触发过等待。
1. **spill 期间新落的 entry 直接进 host。** 实测 spill 出去 127 条、restore 回来 **142** 条——多的 15 条是训练相位期间预取新写的，它们没有在训练时把显存重新吃回去。**这条守卫不是假想的。**

zmq 路径上两个端点都是显式 no-op——尤其 restore，无脑把 host cache 传上 GPU 就正好做反了。

三次 spill 实录：

```
22:06:21  spill:   moved=0/0     entries (0 MiB)       ← 启动时，缓存还空
22:06:26  restore: moved=0/0     entries (0 MiB)
22:15:28  spill:   drained=117   moved=127/127 (7422 MiB) → host
22:25:15  restore: moved=142/142 entries (8145 MiB)    → cuda:0
22:34:28  spill:   drained=92    moved=116/116 (8189 MiB) → host
```

### 1.3 补上 mooncake 路径的 `observe_model_forward`

上游只在 `_encode()` 里记这个指标，所以 mooncake 上 `encoder_model_forward_seconds` **恒为 0**。而它正是当初查出「缓存被 batch 路径绕过」的唯一信号——日志里一个字都没有。缺失的位置恰好是最需要它的那条路径。

### 1.4 回归测试

`scripts/debug/epd_minimal_repro.sh` 从 8 项扩到 **11 项**，两个 backend 都跑通：

- case 9：缓存确实服务重复请求（cold 0 hits/64 → warm 64/64）
- case 10：mooncake 上 spill 真的还回显存（moved 2427 MiB，GPU0 4819 → 1429 MiB）；zmq 上是显式 no-op
- case 11：spill→restore 后缓存**还是缓存**（8/8 命中、0 次 ViT）——一个「丢掉而不是搬走」的实现会通过 case 10 但挂在这里

**写测试时抓到自己一个假阳性**：case 9 原本断言 `forwards == 0`，但那个指标在 mooncake 上根本不存在，**恒真**。正是它该抓的那类 bug 的形状。改成断言 hit/total 计数器后两个 backend 都成立。

______________________________________________________________________

## 2. 今天踩的坑

### 2.1 🔴 mooncake RPC 端口撞 Relax 端口分配器（第一次实验被它挂死）

**必须**在用 mooncake 时钉死端口段：

```bash
export MC_MIN_RPC_PORT=19000  MC_MAX_RPC_PORT=19999
export MC_MIN_PRC_PORT=19000  MC_MAX_PRC_PORT=19999   # 两种拼写都认，实测各自单独生效
```

**为什么是必然不是运气**：mooncake 的 `TransferEngine.initialize(hostname, "P2PHANDSHAKE", ...)` 里 hostname 不带端口，端口由 mooncake 自己挑，实测默认落在 **15303**；而 Relax 的引擎端口分配器基址正是 `find_available_port(15000)`。两个分配器同一个起点。

**为什么会抢到**：Relax 只是「找到一个空闲端口号并记下来」，真正 bind 要等 sglang 的 uvicorn。mooncake 在 scheduler 子进程里初始化，正好落在这个窗口内。

**现象**：

- 每 2s 刷 `readString: too large length from socket: 7018130145941931335` —— 那个整数是 ASCII 的 **`GET /hea`** 被当成长度前缀读，即 `_wait_server_healthy` 的 `/health_generate` 轮询（`sglang_engine.py:348` 的 `time.sleep(2)`）打在了 mooncake 的握手 socket 上
- `[engine-init-barrier:actor] ready 1/2, still-waiting ranks=[0]` 永久卡住
- **最骗人的地方**：那个引擎的 `/health` 一直返回 200（uvicorn 起在别的端口上），查健康检查完全看不出问题
- 竞态，每次挂的引擎不同、也可能不挂 ⇒ 只能钉死，不能靠重试

**定位手法**：`py-spy dump --pid <卡住的 actor> --locals` 读出 `init()` 的 `port` 实参（15364），psutil 看谁在 LISTEN（`sglang::scheduler_TP1`，不是 uvicorn），日志里 grep `<host>:<port>` 会看到它作为 mooncake session_id 出现。

**根因其实在 Relax 侧**：分配端口号却不占住 socket，任何在窗口内调 `get_free_port()` 的第三方都能偷走。真修要改 `_allocate_rollout_engine_addr_and_ports_normal` 让它持有 socket 到 sglang bind 为止——那是 launcher 逻辑，属于 CLAUDE.md 的 Ask First 范围，**没做**。

同类前科：`--encoder-bootstrap-port` 默认 8997 被两个 language 引擎同抢，那个已经在 Relax 里加了第 5 种端口类型解决。

### 2.2 预取「早发」了但 7 分钟后才真发出去：串行 PNG 编码

| | rollout 窗口 | `submit()` | 首个 `/encode` 到达 | 最后一个 |
| --- | --- | --- | --- | --- |
| step 0 | 22:06:26–22:15:28 | 22:07:11（45s 进） | **22:13:21（415s 进）** | 22:15:52（**晚于 rollout 结束 24s**） |
| step 1 | 22:25:15–22:34:28 | 22:25:31（16s 进） | **22:33:17（482s 进）** | 22:34:57（**晚 29s**） |

预取总耗时 520.6s / 565.5s，其中 POST 那一段只有 151s / 100s ⇒ **70-82% 的墙钟时间花在发第一个请求之前**。

`encoder_prefetch.py:135-136`：

```python
for images in _iter_group_images(groups):
    payloads.extend(encode_image_for_rollout_engine(image) for image in images)
```

`encode_image_for_rollout_engine` 做的是 `image.save(buffer, format="PNG")`（`processing_utils.py:404`）——**无损 zlib，对 ~110 张文档扫描件，单线程串行跑完才开始发**。只有 POST 进了 `ThreadPoolExecutor`，编码没有。算下来 3.4s/张，和 PNG 压缩一张 2000×2800 的量级对得上（`ray.get` 那段按 step 0 的冷取只值 ~37s，剩下 ~330s 都是编码）。

**改起来是三行的事**（把编码挪进同一个池），**但改了也没用**：step 1 已经拿到 93.5% 命中率了，恰恰因为中间隔着 9 分钟训练，预取再晚也早就完成。**晚不晚不是瓶颈，缓存热了也没用才是。**

> py-spy 现场确认没跑成（等 step 2 窗口时实验已停）。上面是算术推断 + 代码路径，不是直接观测。

### 2.3 缓存已经贴着上限

第三次 spill 报 `8189 MiB`，而 `SGLANG_VLM_CACHE_SIZE_MB=8192`。要同时装下「本步 + 预取的下一步」，8 GiB 是临界值，已经在 LRU 边界上。**现在 spill 能把它挪走了，训练期不占显存，所以调大是可行的**——但既然缓存命中率对 `rollout_time` 没有影响，调大也没有意义。

______________________________________________________________________

## 3. 哪些留着有用，哪些别再投入

**留着（与 EPD 无关，独立有价值）：**

| 项 | 谁需要 |
| --- | --- |
| mooncake 端口钉死（§2.1） | 任何在 Relax 里用 mooncake 的场景，不限 EPD |
| sglang EPD 的 `input_ids` 支持（bug 1/2，见设计文档） | 任何 rollout 走预展开 `input_ids` 的 RL 框架 |
| `--enable-prefix-mm-cache` 对图片请求生效（bug 4） | 任何开这个 flag 的人——它原本对图片**完全无效**且无任何报错 |
| `mm_cache` 移植进 `encode_with_mooncake` + `observe_model_forward` | 同上，mooncake 侧 |
| spill/restore 机制本身 | 任何「colocate 下有 GPU 常驻缓存要熬过训练相位」的场景 |
| `epd_minimal_repro.sh`（11 项，双 backend） | 上面所有项的回归网 |

**别再投入：**

- 给 encoder 加卡 / 调缓存大小 / 优化传输 —— 都在 0.68% 那个上界之下
- 修 §2.2 的串行 PNG —— 已证明不在关键路径
- mooncake MR 生命周期优化（注册一次而不是每次命中都注册）—— 同上
- 迁到 v0.5.17 的 EPD 部分 —— 除非另有原因需要 EPD

**如果真要提 `rollout_time`，该看的是 generation 侧**：`generate_time` mean 139.5s / max 427.8s，长尾是 3 倍。这才是 525s 里的主体。

______________________________________________________________________

## 4. 复现方式

```bash
# EPD 实验（已含 mooncake 端口修复）
bash examples/on_policy_distillation/mopd/run-mopd-qwen35-9b-8xgpu-colocate-epd.sh

# 不带 Relax 的 sglang EPD 一致性 + 缓存 + spill/restore 回归（需 7 卡）
bash scripts/debug/epd_minimal_repro.sh                    # zmq 臂
BACKEND=mooncake bash scripts/debug/epd_minimal_repro.sh   # mooncake 臂

# CPU 单测
PYTHONPATH=/root/Megatron-LM pytest tests/backends/sglang/ tests/distributed/ray/ -q
```

sglang 侧改动全部落在 `docker/patch/sglang/v0.5.15.post1.patch`（33 文件，三重校验：对 pristine `git apply --check` 通过 / 应用后与工作树逐字节一致 / 文件数不变）。
