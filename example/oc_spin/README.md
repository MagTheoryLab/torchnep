# GPUMD c516 磁势

使用 [nep.in](nep.in) 训练：

```python
from torchnep import train_nep
train_nep("nep.in", "train.xyz", output_dir="output")
```

唯一的磁模型是 `model nep_spin`、`spin_version 1`，对应 GPUMD-MagTheoryLab
`c516` 的 `fd358023c51239d7290d048928b3a0b298451993`。默认 C3/L2/O2/SOC，
本例有 30 个结构坐标和 34 个磁坐标。非 SOC 同配置有 31 个磁坐标。
`spin_compress` 为 1–9，`spin_l_max` 为 0–4，`spin_order` 为 1–4；
受 96 个磁坐标、199 个总坐标限制。

角向结构系数在 spin cutoff 下重新求值，再经共享径向混合、路径权重和中心
primitive 混合构建张量密度。SOC 使用实 CG 耦合；非 SOC 保留空间与自旋
乘积扇区。输出包括局域偶数阶标量、偶标量密度以及各扇区的归一化平方范数。
阶数控制 primitive 的自旋多项式次数，不是能量的体阶，也不保证通道前缀。

`spin_type Fe` 表示只有 Fe 有独立自旋；其它元素输入自旋按零处理，仍提供几何
环境。`spin_basis_size` 默认跟随结构角向 `basis_size`，只能设置相同值。
`spin_cutoff` 默认跟随结构径向 cutoff，可按元素设置，不能超过结构径向 cutoff。
零自旋时保留 degree-zero 几何通道；自旋导数仍有限。

结构和磁坐标进入同一个元素条件化标量能量 ANN，F、virial、mforce、torque
都是它的导数。训练从第一个 epoch 起使用全部配置；Stage 2 和响应组是训练
语义。磁 scaler 为 `1/max(RMS, range, 1)`，不平移，且不使用全一系数的旧
`use_gpumd_qscaler` 路径。TorchNEP 保留自己的优化器，不复现 GPUMD 的 SNES 轨迹。

普通和磁模型均导出 `nep / format_version 1 / model nep[_spin]` 及
`@ parameters`、`@ q_scaler` 块，结构系数按 type-pair-first 排列。只接受这种新格式；
旧普通 `nep4[_zbl]` 和旧磁 `nep4_spin2/3` 文本均被拒绝。`.pt` 微调要求当前
`torchnep-c516-v1` 架构签名；续训还要求完整的 seed、scheduler、数据 lineage 和
训练日程，不补缺省值，也不迁移旧 DDP 前缀、课程日程或缓存。`spin_curriculum 0`
同样报错。`.pt` 为 TorchNEP 训练状态，不是 GPUMD 的 SNES restart。
针对旧描述符的 CUDA 实验、图缓存和版本选择已删除。CPU 上已核对独立冻结
oracle；CUDA 执行与吞吐应在目标设备另验。

## 受控响应结构

在 `nep.in` 中设置 `lambda_spin_response 0.1` 启用响应训练（默认 0，
关闭时不读取响应文件）。非零时必须在 `train.xyz` 同目录提供
`response.xyz`，不使用 `response_file` 关键字。文件只包含完整的
rotation 组，需有 species、位置、晶胞、spin、mforce 以及下述组和
角度元数据。与完整磁性训练数据一致，energy、force、spin、mforce
必须提供且数值有限，只有 virial 可省略。当前专门响应损失使用
磁力生成元；完整提供 E/F 标签是数据契约，不表示已启用组内能量差
或力的专门响应损失。

普通数据仍按 `batch` 做 minibatch；响应按完整组独立调度，沿用
组间等权、每 epoch 平衡和均值加中心化 Huber 目标。无需全训练集
full-batch。即使同一帧也在 train.xyz 中，response 路径不会再加一遍
普通 E/F/MF 损失。baseline 和描述符 scaler 仅由普通训练数据拟合。

响应目标仍仅在 Stage 2 启用，要求 `stage2 1`。若使用验证集，验证
文件需包含独立的 rotation 组供 best-model 选择；不能把训练扫描
复制到验证集。同组或相同结构与自旋的重复帧会被拒绝。使用
`valid_ratio` 时也需预先设计响应组的独立验证，程序不会自动拆分
response.xyz。响应文件 SHA256 记录进 checkpoint，修改文件后不能
静默续训。以前仅在 train.xyz 内标注组而未设显式权重的输入，现在
不会自动启用响应训练。

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
