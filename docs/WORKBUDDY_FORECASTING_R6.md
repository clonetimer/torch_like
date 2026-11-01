# Workbuddy Forecasting R6 — Value Triage / 剩余候选价值筛选

## Outcome / 本轮结果

中文：**R5剩余57项已全部完成价值与任务角色筛选；本轮新增模型0项。**
结论为3项优先深审、32项暂缓、8项当前增益较低不添加、14项不作为独立
预测骨干收录。后两类共22项退出本轮骨干实现队列，但保留理由及重开条件。
不是57项“已实现”，也不是57项正式准入。未解决的许可、源码或接口问题
不等于算法价值低。优先深审只是研究资源顺序，不是实施批准。

English: All57 R5 remaining names have engineering value/task-role dispositions.
**Zero models are added in R6.** Three are research priorities,32 held,8 skipped
for low present incremental value, and14 routed outside the standalone backbone
queue. The22 exclusions remain traceable and reopenable;35 items remain on the
priority/hold side. None of these states is a formal policy score or admission.

| Triage state / 初筛状态 | Count / 数量 | Meaning / 含义 |
|---|---:|---|
| `review_next` | 3 | Next source/contract/admission research, not implementation approval / 优先研究，不代表批准实现 |
| `hold` | 32 | Permission, assets, evidence, native feasibility or task contract unresolved / 许可、资产、证据、可行性或契约未解决 |
| `skip_low_value` | 8 | Present scope-relative return does not justify integration cost / 当前范围内增益不足以支撑投入 |
| `out_of_backbone_scope` | 14 | Training, composition, calibration, retrieval or global classical fitting / 转训练、组合、校准、检索或全局经典拟合方向 |

Mapping remains **39/96 mapped,57 unmapped**. Catalog remains **171 total,
49 forecasting**, public API928, grandfathered baseline161. The historical33 name
matches are still not fidelity certifications. Formal candidate scores, runtime
maturity, Stable, dependencies, version and workflows are unchanged.

中文：筛选覆盖率与模型覆盖率分开计数；排除一个条目不会提高39/96的映射率。
本批没有下载权重、训练数据或上游源码到运行包，没有把校准层、检索器或
随机初始化的基础模型外壳冒充论文所宣称的完整模型。

English: Review coverage and implementation coverage are different denominators.
An excluded name is not a new catalog match. No weights, datasets or upstream
implementation files were installed/vendored; no wrapper inherits a paper's
zero-shot, calibration or benchmark claims.

See the [57-item machine-readable record](../governance/workbuddy_forecasting_triage_r6.json),
[design](superpowers/specs/2026-11-01-workbuddy-r6-triage-design.md) and
[execution plan](superpowers/plans/2026-11-01-workbuddy-r6-triage.md).

## Why these three next / 三项优先方向

| Candidate / 候选 | Missing capability / 增量能力 | Next gate / 下一关 |
|---|---|---|
| DeepState | RNN-conditioned linear Gaussian state-space forecasting and analytical filtering / 可解释状态结构与 Kalman 概率预测 | Formal evidence/scoring, seasonal and covariate contract, posterior/covariance stability, correlated path sampling / 正式证据评分、季节及协变量、后验稳定性和路径采样 |
| Paloma | Lightweight phase-conditioned direction/amplitude residual modulation / 轻量相位条件方向与幅值调制 | Independent evidence plus cycle-index and normalization semantics / 独立效果证据、周期索引和归一化语义 |
| SEER | Patch-quality-aware enhancement/replacement under corrupted observations / 受损 patch 的识别、增强和替换 | Pinned core, controlled robustness evidence and faithful routing/selection / 固定源码、受控鲁棒性证据及路由选择机制 |

English: This ordering values distinct reusable capability, not the most recent
paper or its claimed SOTA. DeepState is not a generic recurrent Gaussian head:
the pinned GluonTS network conditions LDS parameters on covariates and updates
the latent posterior from observations. Paloma is a genuine standalone forecaster
as well as a reusable modulation idea. SEER's reported corrupted-data robustness
does not establish arbitrary NaN/missing-mask API support.

中文：这三项仍需按现有六项硬门槛及证据评分推进，没有预授予分数。
Paloma和TStruct的本次检索未找到可核实的独立重跑；这不证明外部实验不存在。
即使满足其他条件，只有作者论文及官方实现也不能当作A/B级独立证据。
按当前政策，C级系数.75的理论上限75仍低于核心线80；观察状态也不是自动
进入实现的许可。本批不为了实施而抬高主观分数。

English: No grade or score is assigned. We located no verified independent rerun
for Paloma/TStruct in this bounded audit. Author tables, acceptance and code
availability do not create independent evidence. Under the unchanged policy,
C's .75 multiplier caps even a100-point value score at75 before risk, below
core80. Observation is not automatic implementation approval.

## All57 dispositions / 完整57项

中文：下表日期是文档发现日期，不是论文发表日期。点击名称查看对应一手
论文/发布来源；完整来源、已核实commit、许可状态与英文理由见JSON。
`null`表示本次未核实；指向main/master的链接不是不可变源码审计。
两个对照模型用于判断新增机制与投入，并非已经执行的本地精度对比。

English: Dates below are report dates, not publication dates. Primary links
resolve identity; the JSON retains source revisions when inspected, permission
status, comparisons and reopen conditions. Null revisions mean unpinned, not
missing code. Peer comparisons are engineering judgments, not benchmark results.

### June / 六月 (11)

| Candidate / 报告日期 | Disposition / 结论 | Existing peers / 对照 | Reason and reopen condition / 理由与再开条件 |
|---|---|---|---|
| [Z-GCNETs](https://proceedings.mlr.press/v139/chen21o.html) · 06-11 | 暂缓 / Hold | STGCN, STGODE | 拓扑持久图有独特能力，但需要离线特征、时间边界及节点对齐协议；有真实拓扑用例后再评估。 |
| [STSGCN](https://ojs.aaai.org/index.php/AAAI/article/view/5438) · 06-13 | 暂缓 / Hold | STGCN, STFGNN | 完整源码树未见许可声明；先解决权限，再比较同步局部图相对既有图模型的增益。 |
| [DA-RNN](https://www.ijcai.org/Proceedings/2017/366) · 06-14 | 暂缓 / Hold | TFT, TimeXer | 报告链接是社区实现且未见许可；先明确目标/历史驱动变量与多步输出契约。 |
| [DeepGLO](https://papers.neurips.cc/paper_files/paper/2019/hash/3a0844cee4fcf57de0c71e9ad3035478-Abstract.html) · 06-14 | 暂缓 / Hold | FCGAGA, DeepAR | 全局矩阵因子与交替拟合不是普通窗口前向；需要集合级 fit/forecast 及持久状态设计。 |
| [DeepState](https://papers.neurips.cc/paper_files/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html) · 06-16 | 优先深审 / Review | DeepAR, ProTran | 补充 RNN 条件化线性高斯状态空间与 Kalman 概率预测；先完成正式准入和数值/协变量契约。 |
| [STRIPE](https://proceedings.neurips.cc/paper/2020/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html) · 06-16 | 暂缓 / Hold | TimeGrad, ProTran | 形状/时间多样性与 DPP 训练、采样协议不能被普通预测头替代；许可和概率接口待核。 |
| [ST-MetaNet](https://www.kdd.org/kdd2019/accepted-papers/view/urban-traffic-prediction-from-spatio-temporal-data-using-deep-meta-learning) · 06-17 | 低增益不加 / Skip | AGCRN, GMAN | 地理属性驱动的元参数确有区别，但当前无具体元数据用例，移植和输入负担相对既有图覆盖收益偏低。 |
| [TRMF](https://papers.neurips.cc/paper_files/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html) · 06-18 | 非骨干方向 / Reroute | FCGAGA, DLinear | 这是有效的经典全局拟合算法，不是神经窗口骨干；需单独的经典预测/矩阵拟合范围及许可审计。 |
| [DSARF](https://ojs.aaai.org/index.php/AAAI/article/view/16907) · 06-19 | 暂缓 / Hold | ProTran, Koopa | 切换状态与变分因子有独特价值；源码许可、滞后状态推断及采样契约未解决。 |
| [ST-Norm](https://doi.org/10.1145/3447548.3467330) · 06-23 | 非骨干方向 / Reroute | GraphWaveNet, NonstationaryTransformer | 贡献是可组合的时空归一化模块；不另造骨干名称，组件复用也仍需解决源码许可。 |
| [CoST](https://arxiv.org/abs/2202.01575) · 06-26 | 非骨干方向 / Reroute | Autoformer, FEDformer | 对比表征预训练后另拟合回归器；需要两阶段训练/组合接口，不把编码器加任意头称为 CoST。 |

### July / 七月 (16)

| Candidate / 报告日期 | Disposition / 结论 | Existing peers / 对照 | Reason and reopen condition / 理由与再开条件 |
|---|---|---|---|
| [Time-LLM](https://arxiv.org/abs/2310.01728) · 07-02 | 低增益不加 / Skip | PatchTST, ITransformer | 核心依赖冻结预训练语言模型；在当前原生、无强制外部权重范围内投入产出低，不能删掉 LLM 后沿用名称。 |
| [TEMPO](https://arxiv.org/abs/2310.04948) · 07-02 | 低增益不加 / Skip | Autoformer, TimeMixer | 主要价值来自预训练与提示域适应；仅移植分解部分与现有能力重叠，待明确预训练需求再开。 |
| [MOIRAI](https://arxiv.org/abs/2402.02592) · 07-02 | 暂缓 / Hold | PatchTST, DeepAR | 任意变量、掩码、多 patch 和概率混合有真实缺口；任务契约、训练语义与资产范围较大。 |
| [Time-MoE](https://arxiv.org/abs/2409.16040) · 07-03 | 低增益不加 / Skip | ITransformer, DLinear | 稀疏专家和规模预训练成本高；小型随机初始化版本不能继承零样本收益，当前投入增益偏低。 |
| [Timer-XL](https://arxiv.org/abs/2410.04803) · 07-03 | 暂缓 / Hold | PatchTST, TimeXer | 长上下文因果多变量生成值得保留；需真实 TimeAttention、位置/变量语义及滚动泄漏测试。 |
| [TimeFilter](https://proceedings.mlr.press/v267/hu25ac.html) · 07-03 | 暂缓 / Hold | PatchTST, MTGNN | 报告的论文 CC BY4.0 不等于代码许可；先取得明确源码授权，再审逐 patch 图筛选。 |
| [SEMPO](https://arxiv.org/abs/2510.19710) · 07-04 | 暂缓 / Hold | FITS, FEDformer | 轻量参数不等于无需预训练；先界定数据域提示、原生训练和频谱路由相对现有方法的价值。 |
| [xLSTM-Mixer](https://arxiv.org/abs/2410.16928) · 07-04 | 暂缓 / Hold | TSMixer, TimeMixer | 需忠实而稳定的原生 sLSTM 单元与 CPU 可移植性；不能直接继承上游 CUDA 路径的效率结论。 |
| [SKOLR](https://arxiv.org/abs/2506.14113) · 07-04 | 暂缓 / Hold | Koopa, FiLM | 深入审计发现混合许可、缺失 runner 导入及边界风险；独立结果未显示稳定增益，已降为暂缓。 |
| [Dynamic-TMoE](https://arxiv.org/abs/2605.20678) · 07-05 | 暂缓 / Hold | NonstationaryTransformer, Koopa | 专家增删、漂移检测牵涉优化器、检查点和训练生命周期，且许可未知；不能用固定专家池冒名。 |
| [SiGMA](https://arxiv.org/abs/2605.31129) · 07-05 | 暂缓 / Hold | TimeMixer, MICN | 真实算子含修正 Bessel 函数，参考实现用 SciPy；先证明原生可微数值等价与成本，而非替换成普通高斯卷积。 |
| [LatentTSF](https://arxiv.org/abs/2602.00297) · 07-05 | 非骨干方向 / Reroute | ITransformer, PatchTST | 潜空间编码/重建/对齐目标增强既有骨干；应进入表征组合方向，且部分文件为非商业许可。 |
| [KUP-BI](https://arxiv.org/abs/2605.19249) · 07-06 | 非骨干方向 / Reroute | DLinear, PatchTST | 训练历史检索与双向启发增强器；需训练集索引、缓存持久化和防泄漏协议，不新增骨干计数。 |
| [PULSE](https://arxiv.org/abs/2605.16793) · 07-07 | 非骨干方向 / Reroute | NonstationaryTransformer, DLinear | 相位增强与 SAM 训练增广是可插拔框架；转组件/训练方向，源码许可仍未知。 |
| [L-Drive](https://arxiv.org/abs/2605.17730) · 07-07 | 暂缓 / Hold | NonstationaryTransformer, Koopa | 潜在变化上下文可能有价值；许可、稀疏发布说明与相对现有非平稳模型的收益待核。 |
| [SEER](https://arxiv.org/abs/2602.00589) · 07-07 | 优先深审 / Review | PatchTST, ITransformer | 优先研究受损 patch 识别/替换的鲁棒性缺口；需独立受控损坏实验，不能声称已支持任意缺失值。 |

### October / 十月 (30)

| Candidate / 报告日期 | Disposition / 结论 | Existing peers / 对照 | Reason and reopen condition / 理由与再开条件 |
|---|---|---|---|
| [TStruct](https://arxiv.org/abs/2610.17864) · 10-19 | 暂缓 / Hold | PatchTST, TSMixer | MIT 已核实；共享时序基和通道头有区别，但仍需独立证据、patch 整除和真实头部混合契约。 |
| [OliO](https://arxiv.org/abs/2610.23345) · 10-19 | 非骨干方向 / Reroute | PatchTST, ITransformer | 自监督预训练结束后移除附加投影/转移模块；下游骨干仍是原模型，转 SSL 训练方向。 |
| [CTIA-ODE](https://arxiv.org/abs/2610.19831) · 10-19 | 暂缓 / Hold | TimeXer, STGODE | 连续时间外生融合有区别；匿名源码无法核实，求解器、观测时间和未来信息边界待设计。 |
| [Paloma](https://arxiv.org/abs/2610.25392) · 10-20 | 优先深审 / Review | DLinear, FITS | 优先研究轻量相位条件残差调制；源码周期索引与非标准归一化需明确，独立效果证据仍不足。 |
| [PathFM](https://arxiv.org/abs/2610.22358) · 10-20 | 暂缓 / Hold | TimeGrad, ProTran | 联合轨迹流匹配与蒸馏不是普通采样头；未核实完整官方发布，预训练和路径接口成本较大。 |
| [TimeES](https://arxiv.org/abs/2609.32384) · 10-21 | 暂缓 / Hold | FITS, TimeGrad | 时变谱幅与概率路径有独特能力；官方树未见许可，训练拟合频率支持及概率接口待核。 |
| [TempoPFN](https://arxiv.org/abs/2510.25502) · 10-21 | 暂缓 / Hold | DeepAR, PatchTST | 官方推理要求 GPU、Triton/FLA 和预训练资产；不是轻量 CPU 原生移植，等待可选依赖及权重契约。 |
| [Timer-M1](https://arxiv.org/abs/2610.11734) · 10-21 | 暂缓 / Hold | ITransformer, TimeXer | 未核实 M1 专属源码/权重；旧 Timer 家族仓库不能充当 M1 发布和许可证据。 |
| [CVTA](https://arxiv.org/abs/2610.25584) · 10-22 | 暂缓 / Hold | ITransformer, TFT | 短期异步预测需要真实时间戳、掩码和查询状态；当前密集窗口接口不能伪装成不规则采样。 |
| [SAGT](https://arxiv.org/abs/2610.25140) · 10-22 | 暂缓 / Hold | GMAN, STGCN | 低秩有向空间混合可能补充规模能力；许可未知，稀疏算子与真实存储/乘法成本待测。 |
| [TimeFT](https://arxiv.org/abs/2610.28997) · 10-22 | 非骨干方向 / Reroute | PatchTST, ITransformer | 这是频域 tokenizer，不是独立预测器；未来按有许可的通用层研究，不另注册包装骨干。 |
| [Meta-iLaD](https://proceedings.mlr.press/v306/ye26p.html) · 10-23 | 低增益不加 / Skip | Koopa, ProTran | 主要是少样本环境识别与动力系统元学习；当前通用长时序任务收益有限，待专门动力学需求再开。 |
| [DDCI](https://arxiv.org/abs/2610.26987) · 10-23 | 非骨干方向 / Reroute | DeepAR, DLinear | 在线区间校准需要逐步揭示标签和反馈状态；转不确定性校准工具，不作为预测骨干。 |
| [CRC](https://arxiv.org/abs/2512.22428v3) · 10-26 | 非骨干方向 / Reroute | DLinear, PatchTST | 冻结预测器的残差纠正与选择性部署；转校准/纠正组件，v3 版本和来源仍需核实。 |
| [Chameleon](https://arxiv.org/abs/2609.36453) · 10-26 | 暂缓 / Hold | PatchTST, ITransformer | 通道相关状态空间更新有潜力；未核实官方源码及内核可移植性，不能凭占位仓库实现。 |
| [CTLite](https://arxiv.org/abs/2610.26379) · 10-26 | 暂缓 / Hold | FITS, FiLM | 官方仓库仅 README/脚本等，没有模型包/配置且无许可；完整发布后再审真实时间戳/查询接口。 |
| [REMIND](https://arxiv.org/abs/2610.32329) · 10-27 | 暂缓 / Hold | GMAN, TimeGrad | 历史检索支持概率混合，有真实增益方向；许可、FAISS/缓存生命周期与节点对齐成本尚未解决。 |
| [DBCM](https://arxiv.org/abs/2610.32071) · 10-27 | 非骨干方向 / Reroute | DLinear, PatchTST | 冻结模型的测试时适应，实际同时校准输入和输出；转独立流式适应方向，需严格标签时序。 |
| [TimeMagnifier](https://arxiv.org/abs/2610.33470) · 10-28 | 暂缓 / Hold | NHITS, FITS | 粗历史预测细未来是跨分辨率新任务；待源码及物理采样率、细粒度监督和对齐协议。 |
| [PCR](https://arxiv.org/abs/2610.33822) · 10-28 | 非骨干方向 / Reroute | PatchTST, ITransformer | 冻结预训练预测器上的残差专家和闭式路由；属于专门化/训练层，不独立替代骨干。 |
| [STeP-TS](https://arxiv.org/abs/2610.35128) · 10-28 | 低增益不加 / Skip | MICN, ModernTCN | 浅层共享卷积与剪枝有区别，但现有卷积覆盖充分；无明确边缘部署及匹配成本收益时不添加。 |
| [Timestral](https://arxiv.org/abs/2610.37585) · 10-29 | 暂缓 / Hold | PatchTST, TimeXer | 代码 Apache2 与权重 Modified MIT 分开；推理发布不等于完整训练系统，零样本能力不能移植给随机模型。 |
| [RankFormer](https://arxiv.org/abs/2610.36703) · 10-29 | 暂缓 / Hold | PatchTST, Crossformer | Hourglass 注意力与渐进解码可研究，但完整树无许可；先解决权限并核实训练/折叠推理语义。 |
| [GAS-PFN](https://arxiv.org/abs/2610.37598) · 10-29 | 暂缓 / Hold | GMAN, STGCN | 零更新跨路网依赖图先验合成任务与预训练；未核实代码/权重，不能只注册两次注意力调用的外壳。 |
| [TimeH2O-v1](https://h2o.ai/blog/2026/timeh2o-foundation-models) · 10-30 | 暂缓 / Hold | PatchTST, TimeXer | 正式论文和源码仓库尚未核实，权重有门控非商业条款；不接受条款或自动下载，待真实资产/架构证据。 |
| [TimeHAT](https://arxiv.org/abs/2610.36572) · 10-30 | 暂缓 / Hold | PatchTST, TimesNet | 主要增益依赖点/patch 掩码预训练和迁移协议；源码许可未知，优先级低于有明确原生缺口的方向。 |
| [MetaARIMA](https://arxiv.org/abs/2610.34171) · 10-30 | 非骨干方向 / Reroute | DLinear, NBEATS | 学习 ARIMA 配置策略再局部拟合，属于经典拟合/AutoML 配置系统；不新增张量前向模型。 |
| [FARE](https://arxiv.org/abs/2610.40160) · 10-31 | 非骨干方向 / Reroute | TFT, TimeXer | 零售因子/形状检索加误差融合纠正既有预测器；转检索/纠正方向，保留训练开发测试的时间边界。 |
| [IMPACT](https://arxiv.org/abs/2610.40418) · 10-31 | 低增益不加 / Skip | GMAN, TFT | 光伏逆变器、太阳几何、天气和在线校准的专用整合；当前通用库边际收益低，且源码/数据仍待核。 |
| [IMRouter](https://arxiv.org/abs/2610.39246) · 10-31 | 低增益不加 / Skip | ITransformer, PatchTST | 长尾模式路由依赖 LLM、专门训练及分组评估；当前原生骨干范围收益偏低，待具体模式不平衡需求。 |


## Source corrections that changed decisions / 影响取舍的来源修正

### SKOLR: priority downgraded to hold / 从优先项降为暂缓

中文：初审认为线性递推+FFT可直接原生化；固定源码和独立实验深审后改为
暂缓/中优先级。官方脚本实际选用KoopBlock，runner导入缺失模块并强制CUDA；
并行变体有调试断点和未完辅助状态。核心的奇数长度逆FFT、非整除预测长度、
dtype及无约束状态转移需要明确边界处理。它不是“复制几层就可用”的移植。

English: [Pinned source859783c](https://github.com/networkslab/SKOLR/tree/859783c349060f3fc87c9222864d83e9d1747f9d)
selects [KoopBlock.py](https://github.com/networkslab/SKOLR/blob/859783c349060f3fc87c9222864d83e9d1747f9d/layers/KoopBlock.py),
not a canonical SKOLR file. Static inspection found missing runner imports,
unconditional CUDA, unfinished parallel helpers and forecast/FFT boundary risks.
These findings were not executed or repaired. Native implementation remains
possible but its cost and benefit need renewed justification.

The root MIT license does not erase explicit CC BY-NC4.0/patent-reservation
headers in `utils/losses.py`, `utils/m4_summary.py`, `data_provider/m4.py`.
The record therefore says **mixed**, not blanket MIT. No such code is copied.

[TimeES AppendixC.5/Table9](https://arxiv.org/html/2609.32384v2) supplies actual
independent SKOLR comparisons and a [baseline implementation/script](https://github.com/wwy155/TimeES/tree/cdb476741c3b32f76c9474d7351a524b68bee7bf).
Its16 ETT multivariate settings do not show an advantage over Koopa. Configuration
differs from original SKOLR scripts; complete run/tuning provenance was not
audited, so this is not proof of optimal comparative performance. A separate
[energy-domain paper](https://doi.org/10.3390/en19184289) reports useful but mixed
performance. Neither study is a torch_like reproduction.

中文：论文自身的效率附录也不支持“总是更快”；原论文的预测长度协议与
常见长跨度表不同，不能跨报告直接排名。这里选择“暂缓收益验证”，不是
“没有独立实验”或“没有科学价值”。

English: The [original appendix](https://arxiv.org/html/2506.14113v1) also reports
settings slower than Koopa; only branch parallelism is implemented, while
time-axis parallelization is future work. Hold is justified by uncertain useful
increment relative to cost, not by a false claim that independent studies do not exist.

### Permission, completeness and task roles / 权限、完整性与任务角色

- **TStruct:** MIT was verified at [01fd47c](https://github.com/yangzeli/TStruct/blob/01fd47c6f635402e8fc8c9f0cdedd773e6926b0a/LICENSE).
  Earlier retrieval uncertainty is resolved; its hold is now evidence/contract,
  not missing permission. Its head mixes low-rank factors separately before
  multiplication, and patch-square mixing remains dense. / MIT已核实，不能继续写“缺许可”。
- **Paloma:** [def3dfc](https://github.com/TSTS13/Paloma/tree/def3dfc612bd6554be8f2ed7d5174848d6a37119)
  has Apache2.0. Loader and model cycle-index conventions need explicit handling
  when history is not a multiple of the period; source variance uses sample
  correction with statistics in the gradient graph. / 周期索引与归一化不能被通用助手静默替换。
- **CTLite:** the [complete inspected tree](https://github.com/SSMa913/NICLab-CTLite/tree/d324af1207c11f2419aa2aafe620b2a6a5b99e85)
  contains README, .gitattributes and five shell scripts, not the referenced
  model package/configs or a license. / 仓库可访问不等于实现已完整发布。
- **TimeFilter:** a paper's CC BY4.0 does not establish code permission; the
  inspected root/README has no verified code grant. / 论文许可与代码许可分开。
- **Timestral:** [code](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/LICENSE)
  is Apache2.0; [weights](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)
  use Modified MIT with a restriction for companies/employers whose preceding
  month's global consolidated monthly revenue exceeds$20M. This is not ordinary
  unrestricted MIT. No terms were accepted or assets downloaded. / 代码与权重条款不能合并成“MIT”。
- **TimeH2O-v1:** the [vendor/model card](https://huggingface.co/h2oai/timeh2o-v1)
  distinguishes Apache-declared inference code from gated noncommercial weights;
  paper/GitHub were still advertised as coming soon in this audit. / 门控权重和未发布的正式技术资料保留为阻断项。
- **SiGMA:** [Eq3 and appendix](https://arxiv.org/html/2605.31129v1) use scaled
  integer-order modified Bessel functions; the reference names SciPy and a dense
  quadratic regime. A normal Gaussian convolution is not equivalent. / 原生可微数值等价需要先证明。
- **ST-MetaNet:** the report-linked lucktroy repository is a fork; the author
  [panzheyi source and MIT license](https://github.com/panzheyi/ST-MetaNet/tree/b0c2cab1890241a8c4b8d6e4101ca951c748fa9b)
  were separately verified. **DA-RNN**'s report-linked code remains a community
  implementation, not an official author release. / 不把fork或社区版本写成作者官方版本。
- **DBCM:** [primary section3.2](https://arxiv.org/html/2610.32071v1) calibrates
  both input and output around a frozen model, not input only. **ST-Norm** is a
  normalization-module contribution. **TRMF** is a real global-fitting algorithm,
  not a nonexistent model; neither is added to the neural-backbone count here. /
  角色不适合当前接口不等于算法无效。
- **LatentTSF** has NC exceptions despite a root MIT license; **KUP-BI**'s inspected
  MIT file retains an unidentified copyright-holder placeholder. / 精确到文件的许可仍需复核。
- **TempoPFN**'s current NeurIPS2026 status was corroborated; stale EurIPS2025
  search metadata is not an identity mismatch. **CRC** has title/version drift
  requiring a pinned v3 audit, not an established wrong-paper identity. /
  旧索引与版本变化不能当作造假证据。

No universal SOTA, percentage gain, exact weight compatibility or deployment
speed claim is adopted from the reports. DBCM/CRC/FARE results require the actual
sequential correction/calibration protocol, not just inference mathematics.

## Verification and boundaries / 验证与边界

```bash
python tools/validate_workbuddy_triage.py --json
python -m pytest tests/test_workbuddy_triage.py tests/test_candidate_pool.py -q
```

The offline validator checks exact coverage against the historical R5 table,
unique identities, dates, finite integer counts, catalog peers, explicit unknowns,
HTTPS source locations and revision syntax. It rejects unexpected fields, so
formal grades/scores cannot silently appear. Counts derive from the57 records.
The baseline is historical: later catalog additions are allowed without changing
R6 counts. This command does not report the latest post-R6 implementation total.

中文：这些测试证明的是清单与字段契约，不证明论文真实性、许可适用性或
实验效果。初筛包含实际读取原报告、一手身份/摘要、相关官方源码树和许可；
只有明确列出的候选完成更深实现审计。没有声称57篇都做过全文数学审查或
独立数据实验。访问失败仅记为无法核实，不当作内容不存在。

English: Tests validate inventory integrity, not science or legal clearance.
Depth varies: all original reports and primary identities were checked; deeper
source/mechanism audits are explicitly identified. No claim of57 full mathematical
reviews, independent dataset reruns or audited executable external pipelines is
made. A syntactically valid commit/URL still requires human source verification.

Merge gates additionally retain the existing candidate/API/catalog/lazy-index
and AutoML validators, full test coverage>=83%, Ruff, an independent whole-branch
review and quality/DDP/Stable workflows. Actual execution outcomes belong in the
PR; this checklist alone does not assert they have run.

