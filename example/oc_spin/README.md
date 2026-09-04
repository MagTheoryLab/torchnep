# O/C 磁势生产框架

## 模型

O/C 表示两个截断参数：

- O（order）：局域磁集团的相关阶数；
- C（compression）：所有收缩路径共享的径向秩。

新模型的生产默认是 `nep4_spin3` O3-C2-L2-SOC（55D）；本示例通过
`n_max 4 4` 和 `l_max 4 2 0` 配置了 30D 结构描述符。spin3 的
O1/O2/O3 磁通道分别是严格的 9D、43D、55D 前缀。新增的 6 维是
四阶交换的 \(l=1,2\) 键角分辨率，\(l=0\) 块与历史 spin2 精确相同。
提高 C 会统一增加径向分辨率，而不是追加某个指定哈密顿量通道。

每个原子的结构描述符和磁描述符进入同一个元素条件化非线性能量
网络。总能量是唯一的学习输出，原子力、virial、磁力和力矩均由它
求导得到，因此不会引入彼此不一致的独立响应 head。O/C 的全部磁
通道在零磁矩时严格为零；每元素最小二乘能量 baseline 只负责移除
总能量中的成分偏置。

新训练使用 `nep4_spin3`；`nep4_spin2` 的 49D 排布、数值和默认训练
语义只为历史模型保留。角交换的数学定义、无近邻对循环的低成本等价
实现、固定通道表和训练契约见 [NEP4_SPIN3.md](NEP4_SPIN3.md)。
TorchNEP 的 reference/training 路径使用 autograd 导数；只识别
`nep4_spin2` 的下游 reader 不能加载 spin3，必须先补齐 55D reader
并通过同模型、同输入、同精度下的能量及全部导数 parity gate。

## 训练日程

O3-C2 默认启用 O2 到 O3 curriculum；如需从第一个 epoch 直接训练
完整 O3，可在 `nep.in` 设置 `spin_curriculum 0`：

1. 前半段只更新 O2 前缀；
2. 随后线性释放 O3 的 12 个新增输入行；
3. O3 在 Stage 2 前约总 epoch 的 10% 处完全打开；
4. Stage 1 尾段用完整力/磁力目标训练全强度 O3；
5. Stage 2 保留 E/F/V/MF/Tau 监督，不把磁响应权重清零。

600 epoch、Stage 2 从 400 epoch 开始时，spin3 在 epoch 340 完全打开：
1--170 为 O2 warm-up，171--339 线性释放 O3，340--399 用 Stage-1
目标训练完整 O3，400 起进入 Stage 2。课程只缩放 Adam 的实际参数
位移，不再重复缩放梯度；第一次释放 O3 前会只刷新其 q-scaler 并清空
相应 Adam moments。curriculum 开关只改变训练路径，不写入模型格式，
也不增加推理成本。

spin3 的结构块使用带相对分辨率保护的 range scaler；磁块使用不平移的
保零 RMS/range/floor scaler，所以固定磁矩产生的常数底不会在初始化时
把 `tanh` 推入饱和区，近零对称通道也不会被放大。训练入口应保持
`use_gpumd_qscaler=False`；全一径向系数不适用于该描述符。

磁力标签的纵向约定必须显式选择：`spin_mforce_mode full` 拟合完整
\(-\partial E/\partial S\)，`transverse` 用于只定义横向约束场的
定磁矩数据，并按两个内禀自由度统计。\(\tau=S\times m\) 不是独立
head；`lambda_tau` 只是额外的 \(|S|^2\) 横向误差权重。

## 受控响应结构

受控结构仍然需要，但它们属于数据辨识和科学验收协议，不属于
描述符输入，也不是新的模型参数。

- J-like：同一几何下改变相对自旋夹角，并包含 reference 和
  正负小角度分支；
- D-like：同一几何下构造相反旋向的非共线分支，用奇偶组合隔离
  反对称响应；
- 磁各向异性：固定内部自旋拓扑，整体旋转自旋相对于晶格的方向；
- 磁弹和体积响应：对同一组自旋 probe 重复对称的小应变或体积网格，
  用 difference-of-differences 隔离混合导数。

普通独立结构仍是主体，受控组不能替代结构和化学环境多样性。推荐
在 extxyz 注释中使用以下通用字段：

- response_group：同一基础结构及其全部 probe 的组标识；
- response_probe：rotation、chirality、strain 或 volume；
- response_coordinate：角度、应变或体积坐标；
- response_branch：reference、plus 或 minus。
- 每帧使用 DFT 约束收敛后的 `spin:R:3`；训练器按组和坐标自动计算
  `dS/dtheta`，无需在 extxyz 中提供 `spin_tangent`。

使用 valid_ratio 时，训练器按 response_group 整组划分，避免同一
扫描轨迹同时进入训练和验证。最终验收还应保留完全未参与训练的
基础结构或晶胞扫描。

这些结构不要求额外提供 J 或 D 标签。对 rotation 组，三个以上不同
response_coordinate、最终 DFT spin 和 MF 会在 Stage 2 自动启用稳健的
磁力生成元曲线监督：每帧直接计算
sum_i MF_i dot dS_i/dtheta，同时拟合组均值和中心化形状。这既保留
常数响应分量，又避开能量二阶差分的 h^-2 噪声放大；训练仍只使用
同一个标量能量 ANN。

训练仍使用 DFT 的 E/F/MF/Tau；J-like、D-like 和磁弹量由成组预测的
偶/奇响应及导数计算。
当数据没有这些受控方向时，低总 RMSE 不能作为已经学准 J、D 或
磁弹的证据。

当前生产训练器实现了 rotation 生成元曲线监督及整组 train/valid
隔离。chirality、strain、volume 仍作为普通绝对量监督和独立验收；
尚未把它们硬编码成模型输出或专门 loss，因此不能据此宣称所有
J/D/磁弹切片都已自动学准。

## 数值健康证书

`benchmark_production_response_toy.py` 的每个 sealed run 都在
`result.json` 写入 `health`：

- 完整训练分布和 sealed blind 扫描子集各自在 q-scaler 前后的 SVD、
  有效秩、恒定列和条件数；
- O1/O2/O3 增量 sector 各自的 SVD；
- 描述符对位置、自旋和各向同性应变的有限差分 Jacobian SVD；位置
  Jacobian 同时报告原始谱和商去三维整体平移零模后的谱；
- 各 sector 对位置、自旋和应变的导数 RMS，暴露失活通道；
- E 对位置、自旋和应变导数与 F/MF/virial 的独立 FD 闭合；
- 正常、近零和严格零磁矩下的有限性与严格零退化；
- `loss.out` 的非有限数检查。

`analyze_production_response_toy.py` 跨 variant 和 seed 汇总最坏导数
误差、满秩状态、条件数、最小 sector 响应及训练有限性。该证书只
对实际采样的数据分布成立；不能把一次 toy 满秩外推成任意 DFT
数据集都满秩。

响应 benchmark 只有两个语义互斥的 variant：`standard` 使用普通
E/MF 监督，`generator` 在完全相同的 E/MF 数据上额外启用 rotation
生成元监督。`--lambda-m` 和 `--stage2-lambda-m` 显式控制普通磁力
权重，用于检验生成元的定向重加权能否被简单提高全局 MF 权重替代。
健康证书同时报告生成元算子的局部二次度量：在完整 MF 标签已经存在
时，生成元是这些标签的线性组合，严格观测秩增益应为零；其可能收益
只能来自目标响应子空间的定向条件化，不能写成新增标签信息或恢复
严格零空间。
内部的 `--generator-weight` 只用于这项科研 trade-off 扫描；它不是
新的 `nep.in` 用户参数。
当前内部默认权重为 0.1。它来自 noisy toy 的多 seed Pareto 扫描；
`stage2_lambda_m=10` 在该数据上能同时保持普通 MF 与定向响应，但这个
数值依赖标签尺度和噪声，不能作为所有 DFT 数据集的固定默认值。

## 当前不纳入生产面的策略

- 额外线性响应 head 和最小二乘 warm-start；
- freeze/unfreeze 的实验分支；
- 写死 J、D 或某种 DMI 形式的输出 head；
- 未经真实数据 gate 的双网络零磁态 residual；
- 描述符内部 block 开关和 campaign 专用 callback。
