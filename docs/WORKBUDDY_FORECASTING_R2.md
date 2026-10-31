# Workbuddy Forecasting R2: Non-stationary Transformer / Workbuddy 预测第二轮

## Implemented Scope / 已实现范围

中文：基于用户 `Workbuddy-TorchLike/report-2026-06-25.md` 的第 3 个主选模型，本轮将 **Non-stationary Transformers** 作为一个独立、可移植的 PyTorch encoder–decoder 点预测模型收录。与此前已合并的 R1（FITS、STGCN）分开评审、编码及验收，不修改其他 Transformer 的公共 attention API。

English: Formal selection 3 in `report-2026-06-25.md` supplied the canonical Non-stationary Transformers reference. R2 admits a standalone PyTorch encoder–decoder point forecaster and leaves existing shared-attention APIs and other model implementations untouched.

| Field / 字段 | Value / 数值 |
|---|---|
| Registry / 注册键 | `NonstationaryTransformer` |
| Public class / 公开类 | `NonstationaryTransformer`, `NonstationaryTransformerConfig` |
| Category / 类别 | Time-series forecasting / 时间序列点预测 |
| Integration / 实现边界 | `paper_aligned_portable_subset` |
| Literature evidence / 文献证据 | A, two distinct independent studies / 两份独立研究 |
| Admission / 正式准入 | `core, implemented`; score 81 under policy 1.0 |
| Runtime maturity / 运行时成熟度 | `unassessed` |
| Dataset benchmark / 本仓库实测基准 | `unassessed`; no accuracy or latency claim |

## Defining Mathematics / 核心数学机制

中文：每个观察窗口的每个变量沿时间计算平均值 `mu`、总体方差与 `sigma = sqrt(var + eps)`；仅用观察窗口去平稳化。独立的学习式投影器从原始窗口及 `sigma`/`mu` 推断每样本缩放因子 `tau > 0` 和每个编码器 key 位置的 `delta`。投影器用源实现一致的 `Conv1d(seq_len,1,3,circular)`：**卷积长度轴是变量轴，不是时间轴**。向量的计算输入 detach，投影器参数仍可训练。

English: Per-channel population statistics are computed only from the observed window, then inverted after forecasting. Two learnable projectors use detached raw observations and scale/mean statistics to infer positive per-sample `tau` and key-indexed `delta`; source-style circular `Conv1d` treats observed time positions as input channels and variables as the convolution length.

$$
z=(x-\mu)/\sigma,\quad \sigma=\sqrt{\operatorname{Var}_{t,\mathrm{pop}}(x)+\epsilon},
\quad \widehat y = \sigma\odot\widehat y' + \mu.
$$

$$
\operatorname{DSAttn}(Q,K,V)=
\operatorname{softmax}_{k}\!\left(\frac{\tau QK^\top + \mathbf1\Delta^\top}{\sqrt{d_k}}+M\right)V.
$$

中文：编码器自注意力与解码器交叉注意力均使用 `tau + delta`；**解码器自身的因果注意力仅使用 `tau`**。公共掩码使用本库 `True = allow`，与官方源的 `True = block` 语义显式区分。任何掩码把某一 query 的所有 key 全屏蔽时抛异常，避免产生 NaN。

English: Encoder self-attention and decoder cross-attention consume both factors; causal decoder self-attention consumes only `tau`. Public masks mean `True = allowed`, unlike the upstream blocker-mask polarity. Fully blocked query rows raise a clear error.

## API and Portability Decisions / 接口及可移植差异

- `NonstationaryTransformer(**ForecastingConfig fields, factor_hidden=128, eps=1e-5)`; default width 64 and 4 heads; `enc_in == dec_in == c_out`.
- Input `(B,seq_len,C)` and output `(B,pred_len,C)`; `predict(make_forecast_batch(...))`, public factory and save/load are supported.
- The decoder is built from the final normalized `label_len` observations plus **zero future values**. Incoming `x_dec` values, including poisoned true future targets, are ignored. Optional known calendar markers have exactly four columns and can influence the output.
- Local sinusoidal positions and learned value/time projections are a documented portability choice. `log_tau` is clipped to `[-15,15]` before exponentiation; this departs from unbounded upstream values to limit avoidable nonfinite cases.
- Standard `ForecastingConfig` fields `embed` and `freq` are accepted for constructor/factory compatibility but are **not selectors** in this portable subset: both are no-ops, and optional encoder/decoder time markers always use four numeric columns with the same learned projection.
- Dense temporal attention has quadratic window-length cost. No TVM/CUDA extension, extra dependency, pretrained checkpoint, official training-recipe parity or linear-complexity claim.
- No Stable whitelist change or version/release bump. New model evidence stays **`unassessed`** until real local datasets and hardware are benchmarked.

## Test Plan and Actual Scope / 测试范围

中文：模块分为私有注意力核、原始窗口投影器和 encoder–decoder 模型，测试覆盖真实数学算子及使用者可观察的接口。下列文件内的单元、数学和集成测试可独立重跑；本记录未把官方论文数值当作本仓库的测试结果。

English: Separate tests cover exact operator behavior, raw-window factor learning, forecasting boundaries, and public registration. Published paper scores are not treated as local test results.

| Test file / 测试文件 | Main acceptance / 主要验收内容 |
|---|---|
| `tests/test_nonstationary_attention.py` | Literal float64 logit/softmax oracle; key-wise delta, tau, masks, causal isolation and gradients |
| `tests/test_nonstationary_projectors.py` | Circular feature-axis convolution literal oracle; sample/vector outputs; detached inputs and trainable weights |
| `tests/test_nonstationary_transformer.py` | Observed statistics, full forward/backward, config/dtype, future target isolation, known-time markers and causal horizon outputs |
| `tests/test_workbuddy_nonstationary_integration.py` | Public lazy exports, registry, runtime, factory/save-load, grandfathered baseline unchanged |

Run from a configured project environment:

```bash
python -m pytest -q tests/test_nonstationary_attention.py tests/test_nonstationary_projectors.py tests/test_nonstationary_transformer.py tests/test_workbuddy_nonstationary_integration.py
python -m tools.validate_candidate_pool --as-of 2026-10-31
python tools/generate_lazy_exports.py --check
python tools/audit_model_catalog.py --check
python tools/audit_public_api.py --check
python -m pytest -q
```

## Sources and Evaluation Limitations / 来源及评价边界

- Primary / 原论文：Liu et al., [NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html).
- Official pinned code / 官方固定版本：[THUML `c4ec40675d11d50b3d9923657f408d0db6f90f56`](https://github.com/thuml/Nonstationary_Transformers/tree/c4ec40675d11d50b3d9923657f408d0db6f90f56), MIT license.
- Independent 1 / 独立评价一：[TFB, Qiu et al., PVLDB 2024](https://arxiv.org/html/2403.20150v2), **univariate Table 6** Stationary column; its large multivariate Tables 7–8 do not evaluate this model.
- Independent 2 / 独立评价二：[DERITS, Fan et al., IJCAI 2024](https://www.ijcai.org/proceedings/2024/0436.pdf), Table 1 NSTransformer and official-baseline retraining statement; preprocessing, splits and metrics differ from TFB.

中文：证据等级 A 表示论文、官方来源与两份确实评价该模型的独立研究满足政策数量要求，不意味着本库复制了两份试验的数值；现有文献并不支持“所有数据集普遍优于竞品”的说法。原始代码许可证经固定版本核验，当前实现独立编写，且未导入官方数据管线。

English: Literature grade A qualifies source provenance, not reproducibility of published numbers inside this repository. The portable implementation is independently written and does not import the original data pipeline.

## Next Intake / 下一批线索

| Candidate / 候选 | Verified facts / 已核实 | Hold / 暂缓因素 |
|---|---|---|
| FiLM | NeurIPS 2022 Legendre-memory forecaster, pinned MIT author source; TFB includes a genuine independent evaluation | A second 2025 independent electricity study runs a FiLM baseline, but its released code reuses the test block for validation/early stopping; not clean held-out replication. Audit and score separately before implementation. |
| Pyraformer | ICLR 2022 pyramidal temporal graph; pinned Apache-2.0 source; two independent forecasting evaluations | Portable dense masked attention is **quadratic**, while the reference sparse/TVM engine requires additional implementation and asset review. Choose the engine/decoder contract explicitly before claiming speed. |
| ST-Norm | Report links a publicly visible code repository | Explicit source-version license remains unresolved; no code admission from that link alone. |

中文：这 60 项剩余名称中还包含基础模型、表征学习、区间校准及残差修正，不能直接套同一个点预测 `ForecastingModel` 合同。后续必须逐项过硬门槛、评分和数学测试。

English: The 60 remaining report names include foundation/pretrained workflows, representation learners, calibration and residual-correction methods. Each requires task-appropriate admission and acceptance, not an automatic point-forecaster wrapper.
