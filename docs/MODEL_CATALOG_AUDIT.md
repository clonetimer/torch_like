# Model Catalog Audit / 模型目录审计

中文：本文件由 `tools/audit_model_catalog.py` 从实时 registry 确定性生成。接口存在性来自代码检查；成熟度和 benchmark 证据默认保持 `unassessed`，不得据此宣称论文复现或生产可用。

English: This file is generated deterministically from the live registry by `tools/audit_model_catalog.py`. API presence comes from code inspection. Maturity and benchmark evidence remain `unassessed` by default and must not be treated as paper reproduction or production-readiness claims.

## Summary / 摘要

- Registry version / 注册表版本: `0.9.0`
- Domains / 领域数: 13
- Models / 模型数: 171
- Schema-v2 migration ready / 可迁移: 171
- Unmapped integration levels / 未映射值: 0
- Maturity unassessed / 成熟度未评估: 152
- Benchmark evidence unassessed / 基准证据未评估: 152

## Domain Coverage / 领域覆盖

| Domain / 领域 | Models / 模型数 |
|---|---:|
| `forecasting` | 49 |
| `diagnosis` | 2 |
| `detection` | 10 |
| `graph` | 16 |
| `rul` | 7 |
| `sequence` | 19 |
| `vision` | 29 |
| `multimodal` | 5 |
| `generative` | 10 |
| `tabular` | 5 |
| `recommendation` | 6 |
| `audio` | 6 |
| `reinforcement` | 7 |

## Legacy Integration Levels / 旧集成等级

| Value / 取值 | Models / 模型数 |
|---|---:|
| `framework_oriented` | 118 |
| `framework_oriented_dense` | 3 |
| `framework_oriented_lsh` | 1 |
| `paper_aligned` | 8 |
| `paper_aligned_core` | 28 |
| `paper_aligned_dense` | 1 |
| `paper_aligned_mlstm_reference` | 1 |
| `paper_aligned_portable` | 1 |
| `paper_aligned_portable_scan` | 2 |
| `paper_aligned_portable_subset` | 7 |
| `paper_aligned_reference` | 1 |

## API Signals / API 信号

中文：这里只表示类暴露了对应可调用属性，不代表其所有输入、数学性质或任务指标已经验证。

English: These counts indicate callable class attributes only. They do not prove all inputs, mathematical properties, or task metrics.

| Signal / 信号 | Models / 模型数 |
|---|---:|
| `forward` | 171 |
| `predict` | 146 |
| `loss_step` | 73 |
| `fit_step` | 15 |
| `config` | 171 |
| `from_config` | 171 |
| `param_count` | 171 |
| `nn_module` | 171 |
| `task_loss` | 88 |

## Complete Inventory / 完整清单

| Domain | Model | Fidelity | Scope | Execution | Maturity | Benchmark | AutoML | Capabilities | API signals | Migration |
|---|---|---|---|---|---|---|---:|---|---|---|
| `forecasting` | `AGCRN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode, graph_structure_learning | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `AST` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `ASTGCN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `Autoformer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `ConvTimeNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | pure_convolution, patching, multiscale | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `Crossformer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `D2STGNN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode, dynamic_graph | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `DAG` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | exogenous, dual_correlation | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `DCRNN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `DLinear` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | decomposition, linear_baseline, channel_independent | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `DMSTGCN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `DeepAR` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | distribution, sampling, likelihood | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `forecasting` | `FCGAGA` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `FEDformer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | frequency_domain, decomposition | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `FITS` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | frequency_domain, channel_independent, low_pass_filter | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `FiLM` | `paper_aligned` | `subset` | `portable` | `unassessed` | `unassessed` | true | legendre_memory, frequency_domain, channel_independent, multiscale | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `FreTS` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | frequency_domain, complex_mlp, channel_mixing | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `GMAN` | `paper_aligned` | `subset` | `portable` | `unassessed` | `unassessed` | true | fixed_spatial_embedding, future_calendar, spatiotemporal_attention, transform_attention | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `GTS` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode, graph_structure_learning | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `GraphWaveNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `ITransformer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `Informer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `Koopa` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | koopman_operator, frequency_filter, nonstationary_dynamics | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `LogTrans` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `MICN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `MLCNN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `MTGNN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `ModernTCN` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | large_kernel, patching, variable_mixing, revin | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `NBEATS` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `NHITS` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | hierarchical_interpolation, multi_rate_sampling, residual_backcast | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `NonstationaryTransformer` | `paper_aligned` | `subset` | `portable` | `unassessed` | `unassessed` | true | de_stationary_attention, learned_attention_factors, causal_decoder, time_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `PPM` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | distribution, sampling, likelihood, parametric_prior | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `forecasting` | `PatchTST` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `ProTran` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | distribution, sampling, likelihood, latent_variable | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `forecasting` | `Pyraformer` | `paper_aligned` | `subset` | `portable` | `unassessed` | `unassessed` | true | sparse_attention, multiscale, bottleneck_construction, direct_forecast | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `SCINet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | multiscale_interaction | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `SOFTS` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | series_core_fusion, channel_mixing, linear_complexity | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `STDN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `STFGNN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `STGCN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | fixed_graph, spectral_graph_convolution, temporal_convolution | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `STGODE` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode, continuous_dynamics | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `StemGNN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | graph_mode | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `TFT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | exogenous, variable_selection, interpretable_attention, gated_residual | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `TSMixer` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `TiDE` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | exogenous, dense_encoder, covariates | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `TimeGrad` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | distribution, sampling, likelihood, diffusion | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `forecasting` | `TimeMixer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `TimeXer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | exogenous | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `forecasting` | `TimesNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | none | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `diagnosis` | `InceptionTime` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | multiscale_convolution, residual_connection, time_series_classification | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `diagnosis` | `ROCKET` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | random_convolution, time_series_classification, fixed_features, linear_classifier | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `AnomalyTransformer` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | association_discrepancy, attention, reconstruction | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `CBAMTCNDetector` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | false | channel_attention, temporal_attention, forecasting, convolution | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `DCDetector` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | dual_attention, contrastive_representation, reconstruction | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `GDN` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | graph_mode, graph_structure_learning, reconstruction | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `MEMTO` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | reconstruction, memory | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `MTADGAT` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | feature_attention, temporal_attention, forecasting, reconstruction | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `OmniAnomaly` | `paper_aligned` | `core` | `standard` | `benchmarked` | `single_dataset` | true | reconstruction, latent_variable, variational, recurrent | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `RTDetector` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | reconstruction, trend | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `TranAD` | `paper_aligned` | `core` | `standard` | `benchmarked` | `single_dataset` | true | reconstruction, self_conditioning, attention, two_phase | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `detection` | `USAD` | `paper_aligned` | `core` | `standard` | `benchmarked` | `single_dataset` | true | reconstruction, dual_autoencoder, unsupervised | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `graph` | `DGI` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, self_supervised, mutual_information, corruption | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `EGNN` | `paper_aligned` | `full` | `dense` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, coordinates, e_n_equivariance, edge_attributes | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GAT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, attention | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GATv2` | `framework_oriented` | `full` | `dense` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, attention, dynamic_attention | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GCN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GCNII` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, deep_gcn, initial_residual, identity_mapping | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GIN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, weisfeiler_lehman | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GNAN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, interpretable, distance_aware | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GPS` | `framework_oriented` | `full` | `dense` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, attention, local_global_mixing | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GraphCL` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | adjacency, graph_task, self_supervised, graph_contrastive_learning, augmentation_invariant | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `graph` | `GraphMAE` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, self_supervised, masked_graph_autoencoding | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `graph` | `GraphSAGE` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, inductive | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `GraphTransformer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, attention | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `Graphormer` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, attention, centrality, spatial_encoding | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `PNA` | `framework_oriented` | `full` | `dense` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, multi_aggregation, degree_scaling | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `graph` | `SpikeGraphormer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adjacency, node_task, graph_task, attention, spiking | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `rul` | `AutoformerRUL` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | none | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `rul` | `InformerRUL` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | none | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `rul` | `LogTransRUL` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | none | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `rul` | `PatchTSTRUL` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | patching, channel_independent, loss_step | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `rul` | `TCNRUL` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | dilated_convolution, loss_step | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `rul` | `THGNN` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | graph_mode | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `rul` | `TransformerRUL` | `framework_oriented` | `full` | `standard` | `benchmarked` | `single_dataset` | true | full_attention, loss_step | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `BERT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | masked_language_modeling, bidirectional_attention, pooled_output, encoder_only | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `Caracal` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | causal_mask, next_token, generation, fft_mixing, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `GPT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | causal_mask, next_token, generation, pre_norm_transformer, language_modeling | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `Hyena` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | next_token, generation, fft_convolution, data_controlled_gating, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `KimiLinear` | `reference` | `full` | `standard` | `unassessed` | `unassessed` | true | next_token, generation, linear_attention, recurrent_state, chunkwise, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `LLaMA` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | causal_mask, next_token, generation, rope, rms_norm, swiglu, grouped_query_attention | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `LRU` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | next_token, generation, linear_recurrence, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `Mamba` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | next_token, generation, state_space, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `Mamba2` | `paper_aligned` | `full` | `portable_scan` | `unassessed` | `unassessed` | true | next_token, generation, state_space_duality, chunkwise, recurrent_state, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `MambaLike` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | next_token, generation, state_space_like, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `Mixtral` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | causal_mask, next_token, generation, rope, rms_norm, swiglu, sparse_moe, top_k_routing | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `Performer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | next_token, generation, linear_attention, random_features, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `Reformer` | `framework_oriented` | `full` | `lsh` | `unassessed` | `unassessed` | true | next_token, generation, lsh_attention, local_attention, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `RetNet` | `paper_aligned` | `full` | `portable` | `unassessed` | `unassessed` | true | next_token, generation, parallel_retention, recurrent_state, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `S4` | `paper_aligned` | `full` | `portable_scan` | `unassessed` | `unassessed` | true | next_token, generation, state_space, dplr, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `T5` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | seq2seq, encoder_decoder, text_to_text, generation | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `TransformerDecoder` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | causal_mask, next_token, generation | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `TransformerXL` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | causal_mask, next_token, generation, segment_memory, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `sequence` | `xLSTM` | `reference` | `mlstm_only` | `standard` | `unassessed` | `unassessed` | true | next_token, generation, matrix_memory, exponential_gating, recurrent_state, long_context | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `vision` | `AlexNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | relu_cnn, dropout_classifier, image_classification, backbone_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `BYOL` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | self_supervised, image_representation, projection_head, predictor, ema_target | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `vision` | `ConvNeXt` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | depthwise_convolution, large_kernel, layer_scale, image_classification, backbone_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `DETR` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | object_queries, set_prediction, transformer_decoder, object_detection | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `DINO` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | self_supervised, image_representation, teacher_student, ema_target, self_distillation | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `vision` | `DeepLabV3` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | atrous_convolution, aspp, segmentation, dense_prediction | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `DeiT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | patch_embedding, transformer_encoder, distillation_token, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `DenseNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | dense_connection, feature_reuse, image_classification, backbone_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `EfficientNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | mbconv, squeeze_excitation, compound_scaling, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `FCN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | fully_convolutional, segmentation, dense_prediction | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `GoogLeNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | inception_block, multi_branch, multi_scale_convolution, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `GroundingDINO` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | open_vocabulary_detection, text_grounding, object_queries, token_alignment, object_detection | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `InceptionV3` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | factorized_convolution, inception_block, multi_branch, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `MAE` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | patch_embedding, masked_reconstruction, self_supervised, backbone_features | forward, predict, fit_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `vision` | `MLPMixer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | patch_embedding, token_mixing, channel_mixing, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `Mask2Former` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | universal_segmentation, mask_classification, mask_queries, semantic_segmentation, instance_segmentation | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `MoCo` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | self_supervised, image_representation, projection_head, contrastive_loss, momentum_encoder, queue | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `vision` | `MobileNetV2` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | depthwise_convolution, inverted_residual, linear_bottleneck, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `RTDETR` | `paper_aligned` | `subset` | `portable` | `unassessed` | `unassessed` | true | object_detection, real_time, end_to_end, multi_scale, query_selection, post_process | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `ResNeXt` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | residual_connection, grouped_convolution, cardinality, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `ResNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | residual_connection, image_classification, backbone_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `SAM` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | point_prompts, box_prompts, mask_prompts, multimask_output, mask_quality | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `SENet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | squeeze_excitation, channel_attention, residual_connection, image_classification | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `SimCLR` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | self_supervised, image_representation, projection_head, contrastive_loss | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `vision` | `SwinTransformer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | window_attention, shifted_window, patch_merging, image_classification, backbone_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `UNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | encoder_decoder, skip_connection, segmentation, dense_prediction | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `VGG` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | small_kernel_convolution, deep_cnn, image_classification, backbone_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `ViT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | patch_embedding, transformer_encoder, image_classification, backbone_features | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `vision` | `YOLO` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | object_detection, single_stage, anchor_grid, objectness, real_time_detection | forward, predict, config, from_config, param_count, nn_module | `ready` |
| `multimodal` | `BLIP` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | image_text_retrieval, image_text_matching, captioning, cross_attention, language_modeling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `multimodal` | `BLIP2` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | image_text_retrieval, q_former, image_to_language_bridge, image_text_matching, captioning | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `multimodal` | `CLIP` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | image_text_retrieval, zero_shot_transfer, dual_encoder, similarity_logits | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `multimodal` | `LLaVA` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | visual_instruction, image_prefix, projector, causal_language_modeling, generation | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `multimodal` | `SigLIP` | `paper_aligned` | `full` | `standard` | `unassessed` | `unassessed` | true | image_text_retrieval, zero_shot_transfer, dual_encoder, sigmoid_loss | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `ControlNet` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | diffusion, latent_space, control_conditioning, zero_convolution, sampling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `DCGAN` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | adversarial_loss, generator, discriminator, sampling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `DDPM` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | diffusion, denoising, noise_prediction, sampling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `DiT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | diffusion, denoising, noise_prediction, sampling, transformer | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `LatentDiffusion` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | diffusion, latent_space, autoencoding, noise_prediction, sampling, conditioning | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `PixelCNN` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | autoregressive, masked_convolution, exact_likelihood, sampling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `RealNVP` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | normalizing_flow, exact_likelihood, invertible, sampling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `RectifiedFlow` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | flow_matching, rectified_flow, velocity_prediction, euler_sampling, conditioning | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `VAE` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | latent_variable, reparameterization, elbo_loss, sampling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `generative` | `VQVAE` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | discrete_latent, vector_quantization, codebook, reconstruction, sampling | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `tabular` | `FTTransformer` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | continuous_features, categorical_features, feature_tokenization, attention, classification, regression | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `tabular` | `NODE` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | continuous_features, categorical_features, oblivious_trees, differentiable_splits, classification, regression | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `tabular` | `SAINT` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | continuous_features, categorical_features, column_attention, row_attention, classification, regression | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `tabular` | `TabNet` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | continuous_features, categorical_features, feature_masks, interpretability, classification, regression | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `tabular` | `TabTransformer` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | categorical_features, contextual_embeddings, attention, classification, regression | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `recommendation` | `BERT4Rec` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | item_sequence, bidirectional_attention, masked_item_prediction, candidate_scoring | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `recommendation` | `DCN` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | categorical_features, continuous_features, cross_network, deep_mlp, ctr_prediction | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `recommendation` | `DeepFM` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | categorical_features, continuous_features, factorization_machine, deep_mlp, ctr_prediction | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `recommendation` | `LightGCN` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | user_item_graph, bpr_loss, full_sort, implicit_feedback, ranking | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `recommendation` | `NCF` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | user_item_interaction, embedding, ranking, implicit_feedback | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `recommendation` | `SASRec` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | item_sequence, causal_attention, next_item_prediction, candidate_scoring | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `audio` | `AudioSpectrogramTransformer` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | spectrogram, patch_embedding, transformer_encoder, classification | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `audio` | `Conformer` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | acoustic_features, attention, depthwise_convolution, ctc_logits | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `audio` | `EnCodec` | `paper_aligned` | `subset` | `portable` | `unassessed` | `unassessed` | true | raw_waveform, audio_encoding, audio_decoding, residual_vector_quantization, bandwidth_control, reconstruction_loss | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `audio` | `HuBERT` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | raw_waveform, conv_feature_encoder, transformer_encoder, masked_prediction, hidden_unit_prediction | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `audio` | `Wav2Vec2` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | raw_waveform, conv_feature_encoder, transformer_encoder, ctc_logits | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `audio` | `Whisper` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | log_mel, encoder_decoder, cross_attention, autoregressive_generation, asr_logits | forward, predict, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `reinforcement` | `ActorCritic` | `framework_oriented` | `full` | `standard` | `unassessed` | `unassessed` | true | policy_logits, value_function, action_distribution, actor_critic_loss | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `reinforcement` | `DQN` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | q_values, greedy_action, td_loss, discrete_actions | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `reinforcement` | `DecisionTransformer` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | offline_rl, return_conditioning, trajectory_modeling, causal_attention, continuous_actions | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `reinforcement` | `DreamerV3` | `paper_aligned` | `subset` | `portable` | `unassessed` | `unassessed` | true | world_model, discrete_rssm, latent_imagination, continuous_actions, actor_critic, symlog, kl_balancing | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `reinforcement` | `PPOActorCritic` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | policy_logits, value_function, ppo_clip, entropy_bonus, discrete_actions | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `reinforcement` | `SACActorCritic` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | continuous_actions, gaussian_policy, twin_q, entropy_regularization, soft_q_loss | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |
| `reinforcement` | `TD3ActorCritic` | `paper_aligned` | `core` | `standard` | `unassessed` | `unassessed` | true | continuous_actions, deterministic_policy, twin_q, target_policy_smoothing, td3_loss | forward, loss_step, config, from_config, param_count, nn_module, task_loss | `ready` |

## Interpretation / 解释边界

中文：`migration_status=ready` 只表示旧 `integration_level` 能无歧义拆分为 schema v2 字段。模型只有在独立证据登记后才能从 `unassessed` 晋升。

English: `migration_status=ready` only means the legacy `integration_level` maps unambiguously to the draft schema-v2 fields. Promotion from `unassessed` requires separately recorded evidence.
