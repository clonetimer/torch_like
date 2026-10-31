# Workbuddy 预测算法接入 R1 / Forecasting Intake R1

中文：本批从 `Workbuddy-TorchLike` 的 36 份文献周报中接入两个经收录评审的模型：FITS 与 STGCN。完整的 96 个正式条目、既有模型映射和未落地候选见[文献清单](WORKBUDDY_FORECASTING_INTAKE.md)；六项硬门槛、证据与评分见[候选池](MODEL_CANDIDATE_POOL.md)。本页说明实际实现边界，不把周报中的性能宣称作为本仓库结果。

English: This batch integrates two admission-reviewed models, FITS and STGCN, from the 36 `Workbuddy-TorchLike` literature reports. The [intake ledger](WORKBUDDY_FORECASTING_INTAKE.md) maps all 96 formal model names; the [candidate pool](MODEL_CANDIDATE_POOL.md) records gates, sources and scores. This page describes implemented behavior, not a replication of performance claims in the reports.

## 接入边界 / Integration scope

| Model | Admission | Implementation fidelity | Mechanism retained | Deliberate boundary |
| --- | --- | --- | --- | --- |
| FITS | 85 / core, literature grade A | `paper_aligned` | low-pass rFFT, trainable complex frequency interpolation, inverse FFT | paired-real parameters; population variance; explicit FFT length and bin cap; no official checkpoint parity |
| STGCN | 82 / core, literature grade A | `framework_oriented` | temporal GLU → Chebyshev graph convolution → temporal ReLU | node-shared channel normalization and direct multi-horizon head; supplied undirected graph only |

中文：文献证据等级与本仓库模型成熟度是两回事。两者的 `maturity` 与 `benchmark_evidence` 均保持 `unassessed`；没有新增真实数据基准、Stable 白名单、发布版本或预训练权重。历史 registry 准入基线的 161 个条目保持不变；新增注册由候选池的 `implemented` 记录授权。

English: Literature evidence grades are not local maturity grades. Both models retain `maturity="unassessed"` and `benchmark_evidence="unassessed"`. This batch adds no real-data benchmark, Stable allowlist entry, release version or pretrained weights. The historical 161-entry admission baseline is unchanged; implemented candidate records authorize the new registry entries.

## 统一使用方式 / Common usage

```python
import torch
import torch_like as tl

common = dict(
    enc_in=3,
    dec_in=3,
    c_out=3,
    seq_len=24,
    label_len=0,
    pred_len=6,
    dropout=0.0,
)
fits = tl.create_model("FITS", domain="forecasting", **common, cut_freq=5)
stgcn = tl.create_model(
    "STGCN",
    domain="forecasting",
    **common,
    adjacency=[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    d_model=8,
    e_layers=1,
    bottleneck_channels=4,
)
x = torch.randn(2, 24, 3)
for model in (fits, stgcn):
    model.eval()
    with torch.inference_mode():
        prediction = tl.as_runtime(model).predict(model.make_forecast_batch(x))
    assert prediction.shape == (2, 6, 3)

# Choose a new destination: save_model writes configuration and weights.
tl.save_model(fits, "fits-model")
restored = tl.load_model("fits-model").eval()
torch.testing.assert_close(fits(x), restored(x))
```

中文：也可直接使用 `tl.FITS` / `tl.STGCN`，或 `FITSConfig` / `STGCNConfig` 配合 `from_config()`；注册名区分大小写。输入是 `[batch, history, channels]`，输出是 `[batch, horizon, channels]`，两者都要求 `c_out == enc_in`，且只读取 `x_enc`。时间标记、decoder 值及 attention mask 是兼容参数，不影响预测；不会读取未来目标。

English: Direct `tl.FITS` / `tl.STGCN` imports and `FITSConfig` / `STGCNConfig` with `from_config()` are also supported. Registry names are case-sensitive. Inputs have shape `[batch, history, channels]`, outputs `[batch, horizon, channels]`; both require `c_out == enc_in` and consume only `x_enc`. Time markers, decoder values and attention masks are compatibility arguments, not prediction inputs.

运行包含一个优化步骤、统一 runtime 推理及保存/重载校验的示例 / Run the one-step optimizer, runtime and persistence example:

```bash
python -m examples.demo_workbuddy_forecasting
```

中文：这是合成信号接口示例，不是模型优劣比较，也不是官方训练配方。

English: This is a synthetic API example, not a model comparison or an official training recipe.

## FITS 参数与数值约定 / FITS parameters and numerics

| Parameter | Default | Contract |
| --- | --- | --- |
| `cut_freq` | `None` → `min(32, seq_len // 2 + 1)` | retained one-sided rFFT bin count, including DC; not Hz |
| `individual` | `False` | share the frequency map across channels; `True` gives each channel its own map |
| `use_norm` | `True` | per-window mean and population variance, with gradients retained |
| `eps` | `1e-5` | finite positive normalization stabilizer |
| `bias` | `True` | complex affine bias, stored as two real parameter tensors |

中文：`seq_len >= 2`，`label_len` 位于 `[0, seq_len]`，`pred_len` 为正整数。输出低频 bin 数为 `min(cut_freq * (seq_len + pred_len) // seq_len, (seq_len + pred_len) // 2 + 1)`；零填充后用显式总长度做 inverse FFT，再按长度比例缩放和反归一化。实/虚参数保留真正复线性映射的交叉项，支持常规优化器与 `.double()`。

English: Require `seq_len >= 2`, `label_len` in `[0, seq_len]`, and positive `pred_len`. The output low-frequency count is `min(cut_freq * (seq_len + pred_len) // seq_len, (seq_len + pred_len) // 2 + 1)`. Zero padding, an explicit inverse-FFT length, length-ratio scaling and inverse normalization follow interpolation. Paired real/imaginary parameters retain the complex cross terms while supporting ordinary optimizers and `.double()`.

中文：输入必须有限，且为与模型同设备、同 dtype 的 `float32` 或 `float64`；FFT 路径禁用低精度 autocast，不宣称原生 half/bfloat16 FFT 支持。各通道不相互混合。`forward()` 仅返回未来，`reconstruct_and_forecast(x)` 返回 `[B, L+H, C]`；调用方可自行施加历史重建与未来联合监督。官方常规实验使用联合监督，官方 `ft` 模式才只监督未来，本例不把 forecast-only 训练称作官方复现。

English: Inputs must be finite float32/float64 tensors on the model's device and in its dtype. Lower-precision autocast is disabled on the FFT path; native half/bfloat16 FFT is not claimed. Channels never mix. `forward()` returns the future only; `reconstruct_and_forecast(x)` returns `[B, L+H, C]` for optional joint history/future supervision. The official ordinary recipe supervises both, while its `ft` mode supervises only the future; the example does not claim recipe parity.

## STGCN 图与时间合同 / STGCN graph and temporal contract

| Parameter | Default | Contract |
| --- | --- | --- |
| `adjacency` | `None`, rejected | caller-supplied dense, finite, nonnegative, exactly symmetric `[N, N]` matrix |
| `d_model` | `32` | hidden channels, at least 2 |
| `e_layers` | `2` | number of spatio-temporal blocks |
| `bottleneck_channels` | `16` | graph-convolution output channels, positive integer |
| `temporal_kernel_size` | `3` | valid temporal kernel width, positive integer |
| `cheb_order` | `3` | Chebyshev term count including identity; `1` is graph-independent |
| `dropout` | `0.1` | finite probability in `[0, 1)` |

中文：每个输入通道对应一个节点的一维观测，不是任意 `[B,T,N,F]` 多特征图输入。时间窗必须满足 `seq_len > 2 * e_layers * (temporal_kernel_size - 1)`。因此默认结构至少需要 9 步历史。固定图被分离、克隆并持久化；禁止负边、有向/非对称图、非有限值，以及转换到模型默认 dtype 后溢出或消失的正边。

English: Each input channel is a scalar observation at one node, not an arbitrary `[B,T,N,F]` graph feature tensor. Require `seq_len > 2 * e_layers * (temporal_kernel_size - 1)`; the default structure therefore needs at least 9 history steps. The fixed graph is detached, cloned and persisted. Negative, directed/asymmetric or nonfinite edges, and positive edges that overflow or disappear in the model's default dtype, are rejected.

中文：图算子为 `2 * D^(-1/2) * (D-A) * D^(-1/2) / lambda_max - I`，`lambda_max` 由对称特征值分解求得。孤立节点的归一化拉普拉斯行列为零；整个拉普拉斯为零时取 `lambda_max=2`，得到 `-I`，不进行跨节点图混合。Chebyshev 递推作用于特征，未用普通一阶 GCN 替代。

English: The graph operator is `2 * D^(-1/2) * (D-A) * D^(-1/2) / lambda_max - I`, with the largest eigenvalue computed from the symmetric Laplacian. Isolated nodes have zero normalized-Laplacian rows/columns. A zero Laplacian uses `lambda_max=2`, giving `-I` without cross-node mixing. Chebyshev recurrence operates on features; it is not replaced with a first-order GCN.

中文：这是稠密图实现：构造时特征值分解约 `O(N^3)`，每块图乘约 `O(B * L * K * N^2 * D)`，另含时间/通道投影成本；没有稀疏 `O(|E|)` 保证。节点共享归一化和直接多步输出头与官方节点专属归一化、单步递推不同，因此明确标为 `framework_oriented`。

English: This is a dense graph implementation: graph setup costs approximately `O(N^3)`, and graph multiplication per block `O(B * L * K * N^2 * D)`, plus temporal/channel projections. Sparse `O(|E|)` behavior is not claimed. Node-shared normalization and a direct multi-horizon head differ from the official node-specific normalization and one-step rollout, hence the `framework_oriented` label.

## 验证记录 / Verification

中文：本地环境为 Linux、Python 3.12.14、PyTorch 2.10.0+cpu、scikit-learn 1.8.0、Ruff 0.17.1。未改动的上游基线 `a403cb7a790136fa5e351a9154030e7a0ece82b9` 全量测试为 **764 passed, 2 skipped, 161 subtests passed**。新增检查用于数学与接口验真，不改变成熟度；本批最终本地结果如下，结构化记录见 [validation/workbuddy_forecasting_r1.json](../validation/workbuddy_forecasting_r1.json)。

English: Local environment: Linux, Python 3.12.14, PyTorch 2.10.0+cpu, scikit-learn 1.8.0 and Ruff 0.17.1. The untouched upstream baseline `a403cb7a790136fa5e351a9154030e7a0ece82b9` passed **764 tests, with 2 skips and 161 passing subtests**. The added checks establish mathematical/API behavior, not maturity. Final local results follow, with a [structured validation record](../validation/workbuddy_forecasting_r1.json).

| Check / 检查 | Result / 结果 |
| --- | --- |
| Full regression, including local CPU/Gloo / 全仓回归，含本机 CPU/Gloo | 878 passed, 2 skipped, 261 subtests passed; 308.29 s |
| Skip reasons / 跳过原因 | Two existing optional LightGBM tests; LightGBM not installed / 两项既有可选 LightGBM 测试，未安装该依赖 |
| Combined statement/branch coverage / 整体语句与分支覆盖率 | 84.1765%, above the unchanged 83% gate |
| FITS / STGCN module coverage / 新模型模块覆盖 | Both 100% statements and 100% branches in this run |
| New model/integration tests / 新模型及集成测试 | 109 passed, 100 subtests passed |
| Lazy-index generator regression / 索引生成器回归 | 5 passed |
| Compile, Ruff lint/format / 编译及代码规范 | PASS; 399 Python files accepted by the CI formatting scope |
| Candidate / example / artifact checks / 准入及协议检查 | PASS; 18 gated, 16 scored, 18 reviews; 11 YAML examples; no saved run directories to validate |
| Registry / public API / 注册表及公共接口 | 167 models, 45 forecasting; 912 protected symbols, exactly 8 additions and no changed/removed old symbols |
| Synthetic optimizer/runtime/persistence example / 合成训练与持久化示例 | FITS and STGCN output `(4, 6, 3)`; finite; reload PASS |
| Full wheel / 完整 wheel | Build + isolated install + registry/config/predict/save/load PASS; imports verified outside the checkout |
| Existing Stable wheel / 既有精简 Stable wheel | Separate build, install, original smoke, content boundaries and coexistence PASS; no new model admitted |

中文：完整包验证版本为 `torch_like 0.9.0`，既有精简包为 `torch_like_algorithms 0.2.0`；这只是从当前源码构建的本地验证包，没有发布或修改版本号。隔离环境通过文本 `.pth` 复用现有第三方依赖，在 `/tmp` 使用 `python -I` 运行，并确认全部已加载的本库模块来自隔离 `site-packages`。Stable smoke 没有导入完整框架；DLinear 的既有 benchmark 证据及证据文件均未改变。

English: Validation builds use `torch_like 0.9.0` and the existing `torch_like_algorithms 0.2.0`; they are local source builds, not releases or version bumps. A fresh environment reuses third-party dependencies through text-only `.pth` paths. Checks run from `/tmp` with `python -I` and verify all loaded package modules come from the isolated `site-packages`. The Stable smoke does not import the full framework. DLinear's benchmark evidence and the evidence registry are unchanged.

- FITS: explicit DFT oracle, complex cross terms, odd/even lengths, cutoff boundaries, normalization, independent/shared channels, input and parameter gradients, optimizer updates, dtype and invalid-input handling.
- STGCN: scaled Laplacian and Chebyshev recurrence, residual time alignment, graph dependence and order-one ablation, node permutation, gradient flow, graph ownership, configuration/state round trips and graph validation.
- Integration: public/lazy exports, registry creation, dataclass configuration, `ModelRuntime`, public save/load, float64 round trips, no dependency on poisoned future decoder values, admission/evidence boundaries.
- No CUDA/NCCL, Windows, official checkpoint parity, real-dataset accuracy or benchmark promotion is claimed.

复查命令 / Recheck commands（假设已按 [DEVELOPMENT.md](DEVELOPMENT.md) 安装开发依赖及 editable package；未安装源码包时，可给 `tools/validate_*.py` 命令加 `PYTHONPATH=.` / assumes the development dependencies and editable package are installed; otherwise set `PYTHONPATH=.` for the `tools/validate_*.py` commands）:

```bash
python -m compileall -q torch_like runner tests examples tools
python -m ruff check torch_like runner tests examples tools
python -m ruff format --check torch_like runner tests examples tools
python tools/generate_lazy_exports.py --check
python tools/validate_candidate_pool.py
python tools/validate_automl_examples.py
python tools/validate_automl_runs.py --allow-empty --skip-legacy runs
python tools/audit_model_catalog.py --check
python tools/audit_public_api.py --check
python -m pytest --cov=torch_like --cov-report=term-missing
python -m examples.demo_workbuddy_forecasting
python -m build --wheel --no-isolation
```

中文：运行包含本机双进程 CPU/Gloo 通信的测试时，执行环境必须允许本机 TCP 通信。覆盖率门槛保持仓库原有的 83%，不禁用原测试、不修改 Stable 包装规则来绕过验收。CI 使用 Python 3.13；本地 Python 3.12 的通过不能代替远端 CI 结果。

English: Tests include two local CPU/Gloo processes and require loopback TCP permission. The repository's 83% coverage threshold and Stable packaging rules remain unchanged; existing tests are not disabled to obtain acceptance. CI uses Python 3.13, so local Python 3.12 success is not a substitute for the remote result.

中文：重建按需导出索引时，同时修复了生成器与格式化器之间的冲突：`--check` 比较 Python AST 的内容与结构，而非引号或换行的字节形式。新增测试确认格式变化可通过，但导出目标改变、语法错误和文件缺失仍会报 stale；检查不会执行或改写索引。默认生成行为不变。

English: Regenerating lazy exports also exposed a generator/formatter mismatch. `--check` now compares Python AST content and structure instead of quote/newline bytes. Added tests accept formatting changes but still reject changed export targets, syntax errors and missing files. Checking neither executes nor rewrites indexes; default generation is unchanged.

## 来源与校正 / Sources and corrections

- FITS: `report-2026-07-01.md`, formal entry 3; [paper](https://arxiv.org/abs/2307.03756), [official Apache-2.0 implementation](https://github.com/VEWOXIC/FITS), [official training recipe](https://github.com/VEWOXIC/FITS/blob/main/exp/exp_main_F.py).
- STGCN: `report-2026-06-13.md`, formal entry 1; [IJCAI 2018 paper](https://www.ijcai.org/proceedings/2018/505), [official BSD-2-Clause implementation](https://github.com/VeritasYin/STGCN_IJCAI-18).
- 中文：6 月 13 日周报误链接到 STSGCN，且图需求描述不准确；本批按原论文和官方 STGCN 仓库校正，未把 STSGCN 当作同一算法。
- English: The June 13 report incorrectly linked STSGCN and misstated the graph requirement. This integration follows the primary STGCN paper and official repository, not that mistaken identity.

中文：实现独立编写；官方许可证与独立评价证据完整列于候选池。候选清单中的其余 61 个名字仍需逐项评审；ST-Norm 的许可门槛未消除，大型预训练模型与辅助校准/表征方法不能自动变成普通点预测模块。

English: Implementations are independently written; the candidate pool records licenses and independent evaluations. The remaining 61 names still require individual review. ST-Norm's license gate is unresolved, and pretrained foundation models or auxiliary calibration/representation methods are not automatically ordinary point forecasters.
