# Workbuddy R7: DeepState 与三项重点候选评审

## 本轮结果 / Outcome

新增原生 PyTorch `DeepState`、`DeepStateConfig`、`DeepStateBatch`，将结构化状态空间推断接入现有预测接口。
DeepState 正式评分 **68，观察级收录**；Paloma **50.25**、SEER **60.5** 均低于65分门槛，本轮保留评审记录。
完整理由、来源版本及逐维评分见 [候选池](MODEL_CANDIDATE_POOL.md#deepstate-neurips-2018)。

This round implements one observation-tier native probabilistic forecaster and
completes the three R6 priority reviews under the unchanged admission policy.
The implemented model has `integration_level=paper_aligned_portable_subset`;
`maturity` and `benchmark_evidence` remain `unassessed`.

| 统计 / Measure | R6 | R7 |
|---|---:|---:|
| 全库注册模型 / All registered models | 171 | 172 |
| 预测模型 / Forecasting models | 49 | 50 |
| 96个报告名称中已映射 / Report-name mappings | 39 | 40 |
| 尚未映射 / Unmapped report names | 57 | 56 |
| 旧版豁免清单 / Grandfathered registry | 161 | 161 |

原57项在R7后的去向：**1项实现、32项暂缓、2项未通过正式评分、8项按当前增益不添加、14项归入非预测骨干方向**。
这些状态合计57项，只有DeepState增加实现计数。R1–R6表格及R6机器记录保留历史含义。

## 功能及输入 / Capability and input contract

DeepState 以共享残差LSTM生成状态空间参数，使用解析高斯滤波更新隐状态。
每个批次行是一条单变量序列；水平、趋势与季节项共同参与预测。
季节索引、已知协变量和静态类别由调用方提供。

| 接口 / API | 返回 / Result | 含义 / Meaning |
|---|---|---|
| `predict(batch)` / `forward(...)` | `[B,H,1]` | 条件预测均值 |
| `predict_distribution(batch)` | `mean`, `scale`, `distribution` | 高斯边际均值及标准差，`scale`不是方差 |
| `sample(batch,num_samples=S)` | `[S,B,H,1]` | 沿同一隐状态轨迹抽样，保留跨时间相关性 |
| `fit_step(batch)` | 标量 / scalar | 逐步吸收未来已观测标签的条件联合NLL，按全批有效标签数平均 |
| `context_nll(batch)` | 标量 / scalar | 各非空序列的上下文平均NLL，再按序列平均 |

`enc_in=dec_in=c_out=1`。`x_enc[B,seq_len,1]`是历史值，
`observed_enc`为同形状bool掩码；省略掩码表示全部已观测。
`y_true[B,pred_len,1]`及`observed_dec`仅用于训练。
已遮蔽的NaN/Inf在算术运算前清零；缺失位置仍推进一个状态步。
**每个张量位置代表一个离散时间步**；索引跳跃不改变状态推进时长。

默认`time_feature_dim=4`需要显式实值特征：
`x_mark_enc[B,seq_len,D]`和`x_mark_dec[B,label_len+pred_len,D]`。
只使用解码器特征的末尾`pred_len`行。`time_feature_dim=0`可省略这些字段。
`seasonal_periods=(p1,...,pK)`需要整数
`seasonal_indices_enc[B,seq_len,K]`及
`seasonal_indices_dec[B,label_len+pred_len,K]`；第k列范围为`[0,pk)`。
配置`cardinalities`及同长度`embedding_dimensions`时，还需
`feat_static_cat[B,len(cardinalities)]`。
`x_dec`可省略，其数值始终不参与预测。所有日期、时区、相位和数据来源语义由调用方确定。

## 使用示例 / Usage

以下随机数据只演示接口，不代表真实遥测或效果评价。
示例明确关闭实值协变量，并提供周期4的离散季节相位。

```python
import torch
import torch_like as tl

torch.manual_seed(7)
model = tl.create(
    "DeepState", backend="torch_like",
    enc_in=1, dec_in=1, c_out=1,
    seq_len=8, label_len=0, pred_len=3,
    d_model=8, e_layers=2, time_feature_dim=0,
    seasonal_periods=(4,), dropout=0.0,
)
history = torch.randn(2, 8, 1)
observed = torch.ones_like(history, dtype=torch.bool)
observed[:, 3] = False
history[:, 3] = float("nan")
batch = tl.DeepStateBatch(
    x_enc=history, observed_enc=observed,
    seasonal_indices_enc=torch.arange(8).remainder(4)[None, :, None].expand(2, -1, -1),
    seasonal_indices_dec=torch.arange(8, 11).remainder(4)[None, :, None].expand(2, -1, -1),
    y_true=torch.randn(2, 3, 1),
)
runtime = tl.as_runtime(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
runtime.loss(batch).backward()
optimizer.step()
model.eval()
with torch.no_grad():
    mean = runtime.predict(batch)                 # [2,3,1]
    marginal = runtime.distribution(batch)        # Gaussian mean/standard deviation
    paths = runtime.sample(batch, num_samples=20) # [20,2,3,1]

# Existing public persistence, in a new caller-selected directory:
# tl.save_model(model, "deepstate_checkpoint")
# restored = tl.load_model("deepstate_checkpoint")
```

`DefaultBatchAdapter`和设备移动保留`DeepStateBatch`扩展字段。
通用`(x,y)`训练数据对适用于`time_feature_dim=0`、无季节项和静态类别的配置；
有额外特征时传入该批次类，或提供显式的批次转换函数。
共享预测批次定义和其他模型的运行时分派维持原有接口。

## 数学与工程选择 / Mathematical and engineering choices

- 发射、观测更新、状态转移按此顺序执行。历史处理返回的状态直接用于首个未来观测。
- 结构创新为单个标量随机扰动乘以发射向量，保留跨结构分量的秩一协方差项。
- LSTM仅消费已知协变量/静态嵌入；目标值及目标缩放量不进入参数网络。
- 采用Joseph协方差更新和对称化。低噪声季节例在float32第8步出现负方差，故高斯代数使用float64累积，历史/未来边界保留该精度，公开输出回到模型dtype。这增加计算和内存成本。
- 均值绝对值缩放仅使用当前行已观测历史并保留梯度；全零或全缺失行使用scale1。训练似然包含原始量纲的对数Jacobian。
- 仅支持float32/64密集张量、Gaussian及dropout0；AMP在模型内部关闭。固定状态转移随模型保存，过滤后验每次重算。
- 采样Cholesky只对失败行增加至多`100*eps*covariance_scale`的舍入级抖动；更严重的非正定协方差明确报错。极端数值仍需缩放。
- 预测标准差转换回模型dtype后仍须严格为正；若极小数据尺度使float32标准差下溢为0，会提示调整尺度或使用float64。

这是原生可移植实现范围。条件未来训练目标、逐行缩放回退、显式日历、零dropout及高斯代数精度均记录为适配选择。
论文数据集精度、区间校准、上游检查点/完整训练流程等价性和CPU/CUDA资源表现尚待实测。

## 三项重点候选的依据 / Priority-review evidence

**DeepState**：VAEneu明确报告重跑GluonTS基线；TAGI-LSTM/SSM正文明确报告为Appendix L新增重跑。
仅计这两处独立作者的实验陈述，TAGI复用既有结果的主表不计入。
Appendix L完整协议及外部执行产物尚未审计；若该新增重跑证据被推翻，保持评分维度不变时将降至B/60.4。

**Paloma**：保留相位条件残差调制的价值判断；目前核实的是作者多种子实验。
首次预测相位与历史模板的索引关系、样本方差归一化和短窗口行为仍需明确。
评分75×0.75−6=50.25，本轮不新增代码。

**SEER**：采用ICML2026预测模型的明确身份。外部日志报告七个数据集上的26个清洁预测结果格，
但原始产物、独立基线和受损数据稳健性尚未充分核实。阈值0同时跳过替换和额外因果编码器，
相关消融无法单独归因于筛选器。即授予B证据，75×0.9−7=60.5，仍未达到收录门槛。

维度分数是维护者对工程价值的判断。Paloma/SEER逐维拆分与风险分项由本轮补全，
保持先前确定的总分/证据/风险不变。三项复审日期设为2027-02-01，新证据可提前触发复审。
所有来源链接、固定版本和不确定性保存在[正式候选记录](MODEL_CANDIDATE_POOL.md)。

## 验证与交付 / Verification and delivery

专门测试覆盖独立稠密联合高斯似然、缺失子集、梯度、边界顺序、缩放Jacobian、
相关轨迹采样、配置/精度保存恢复、公开工厂、训练适配与按需导出。
最终全量测试、覆盖率、安装包验证、独立代码评审及实际PR/main工作流结果在本轮PR中记录。

上游Apache NOTICE随适配源文件和完整wheel保留。精简Stable包范围、版本、依赖及工作流沿用现有配置。
草稿PR#30属于单独的Stable候选分支，本轮未修改。
