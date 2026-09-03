# `nep4_spin3` 描述符与训练契约

`nep4_spin3` 是新的磁性模型格式。`nep4_spin2` 的 49 维
O3-C2-SOC 排布、数值和训练默认值保持不变；两种格式不共享隐式兼容
逻辑。spin3 的默认生产切片为 O3-C2-L2-SOC，共 55 个磁描述符通道，
O1、O2、O3 分别是严格的 9、43、55 维前缀。

## 角分辨四阶交换

记中心原子为 \(i\)，近邻为 \(j,k\)，
\(\hat{\boldsymbol r}_{ij}=\boldsymbol r_{ij}/r_{ij}\)，
\(d_{ij}=\boldsymbol S_i\cdot\boldsymbol S_j\)，
\(w_c(r_{ij})\) 为元素条件化的第 \(c\) 个压缩径向通道。spin3
的数学定义是

\[
C_l^{ab}(i)=\sum_{j\ne k}
w_a(r_{ij})w_b(r_{ik})d_{ij}d_{ik}
P_l(\hat{\boldsymbol r}_{ij}\cdot\hat{\boldsymbol r}_{ik}),
\qquad l=0,1,2.
\]

这里 \(a\le b\)，\(C=2\) 时径向对为
\((0,0),(0,1),(1,1)\)。\(l=0\) 与 spin2 的
`correlation_distinct_neighbor` 逐位相同；\(l=1,2\) 是新增的
6 维，因此 spin3 是严格超集，而不是重新解释旧维度。

上述定义只用于说明物理信息，程序不显式枚举 \((j,k)\)，也不计算
球谐函数。实际计算先在一次边遍历中累积

\[
A_{lc}(i)=\sum_j w_c(r_{ij})d_{ij}
\Phi_l(\hat{\boldsymbol r}_{ij}),\qquad
D^{ab}(i)=\sum_j w_a(r_{ij})w_b(r_{ij})d_{ij}^2,
\]

其中

\[
\Phi_0=1,\qquad
\Phi_1(\hat{\boldsymbol r})=\hat{\boldsymbol r},\qquad
\Phi_2(\hat{\boldsymbol r})=
\operatorname{STF}_5(
\hat{\boldsymbol r}\otimes\hat{\boldsymbol r}).
\]

利用加法定理可精确改写为

\[
\begin{aligned}
C_0^{ab}&=A_{0a}A_{0b}-D^{ab},\\
C_1^{ab}&=A_{1a}\cdot A_{1b}-D^{ab},\\
C_2^{ab}&=\tfrac32 A_{2a}\cdot A_{2b}-D^{ab}.
\end{aligned}
\]

`STF5` 采用正交的五分量无迹对称表示，满足

\[
\tfrac32\Phi_2(\hat{\boldsymbol r})\cdot
\Phi_2(\hat{\boldsymbol r}')
=P_2(\hat{\boldsymbol r}\cdot\hat{\boldsymbol r}').
\]

因此在生产 \(C=2,l_{\max}=2\) 配置下，每条有向边只增加
\(C(3+5)=16\) 个累加量，随后每个中心做 6 个 Gram 收缩。复杂度为
\(O(N_{\mathrm{edge}}C+N_{\mathrm{atom}}C^2)\)（生产的 \(C=2\)
即线性边成本），没有
\(O(\sum_i n_i^2)\) 的近邻对循环。

CUDA 路径在同一次边遍历中生成原 spin2 的 85 维 primitive bank 和
这 16 个角矩，再用一个中心收缩 kernel 形成 6 个新通道。V100、512
原子、每原子 24 个近邻的独立 FP64 oracle 中，descriptor、位置导数、
自旋导数和训练 pullback 的最坏误差分别不超过
`1.42e-14`、`1.42e-14`、`2.84e-14`、`6.82e-13`；相对 spin2 的
前向、一阶导数和训练路径开销为 `5.70%`、`7.20%`、`3.74%`。

## 固定排布

| 阶数 | 通道 | 范围 |
|---|---|---:|
| O1 | `local_s2` 到 `center_l2_environment` | 0:9 |
| O2 | `edge_l0_dot2` 到 `correlation_same_edge` | 9:26 |
| O2 | `correlation_distinct_l0/l1/l2` | 26:35 |
| O2 | 四个 L1/L2 SOC 通道 | 35:43 |
| O3 | `edge_l0_moment_gate` 到 `coupling_l112_edge_response` | 43:55 |

完整逐通道名称、起止位置和宽度由
`MagneticPolynomialDescriptorV3.schema` 定义，并由格式测试冻结。
任何新增信息必须使用新的模型格式，不能在 `nep4_spin3` 内静默改
宽度或换语义。

必须保持的恒等式包括：

- O1/O2 是高阶模型的逐位前缀；
- `correlation_distinct_l0` 是 spin2 对应通道的精确投影；
- 对邻居存储顺序不变；
- 对位置和轴矢量自旋的联合 O(3) 变换不变；
- 对全局时间反演 \(\boldsymbol S\mapsto-\boldsymbol S\) 不变；
- 零自旋严格映射到零磁描述符；
- 三个角通道族都是自旋四次齐次量。

## 径向与缩放

spin3 的初始压缩径向族统一为
\(w_c(r)=(r/r_c)^c f_c(r)\)。生产 \(C=2\) 与 spin2 相同；
\(C>2\) 不再使用不连续的
\(((x f_c)^2,x^3f_c,\ldots)\) 混合约定。

结构块沿用相对分辨率保护的 range scaler。磁块不能减均值，否则会
破坏零自旋极限；它使用纯乘法的保零 RMS/range/floor scaler：

\[
s_q=\left[\max\left(
\operatorname{RMS}(q),q_{\max}-q_{\min},1
\right)\right]^{-1}.
\]

该形式把固定磁矩产生的常数底缩到 \(O(1)\)，用 range 抑制稀疏尾部，
并让任何通道的增益都不超过 1，避免把数据中近零的对称/SOC 通道和
数值噪声一起放大。对用于估计 scaler 的样本，它还给出
\(|q s_q|\le2\) 的严格界。scaler 只乘不平移，所以
\(q(\boldsymbol S=0)=0\) 仍严格成立。

## 训练量的定义

- `spin_mforce_mode` 必须在 spin3 训练输入中显式选择，避免把横向约束场
  静默解释成完整导数。
- `spin_mforce_mode full` 拟合完整的
  \(\boldsymbol m=-\partial E/\partial\boldsymbol S\)，包括纵向
  分量。标签必须采用同一约定。
- `spin_mforce_mode transverse` 只在 spin3 可用，拟合
  \(\boldsymbol m_\perp=\boldsymbol m-
  (\boldsymbol m\cdot\boldsymbol S)\boldsymbol S/
  |\boldsymbol S|^2\)，按两个内禀自由度报告 RMSE；它适合只提供
  横向约束场的定磁矩 DFT 数据。零磁矩点不定义方向，不计入该项。
- \(\boldsymbol\tau=\boldsymbol S\times\boldsymbol m\) 不是独立
  预测。`lambda_tau` 只会按 \(|\boldsymbol S|^2\) 额外加权磁力
  残差的横向部分；spin3 的 torque RMSE 按两个内禀自由度归一。
- SOC 模型的 position-only virial 可以含反对称自旋-晶格转矩，而 DFT
  stress 约束的是对称应变。spin3 的 virial loss 直接用三个对角元和
  三个成对平均的非对角元，不构造临时 3×3 张量。
- O2→O3 课程只缩放 Adam 的实际参数位移，不再同时缩梯度；全局梯度
  裁剪的范数按相同课程系数计入 O3 行，冻结行不会压低 O1/O2 梯度。O3 在
  Stage 2 前总 epoch 的约 10% 处完全打开；第一次释放前仅重算 O3
  scaler，并清空与旧尺度对应的 O3 Adam moments。O1/O2 scaler 不动，
  所以能量连续。
- rotation-response 训练在每个 epoch 内给每个完整 group 相同总权重；
  spin2 保留原 modulo 轮转语义。

## 审核取舍

- 保留新增的 `correlation_distinct_l1/l2`。直接把这六维 scaler 置零、
  其余参数量和训练设置不变的 FeGe 三种子 100-epoch 消融中，响应切片
  的横向磁力测试 RMSE 中位数由 `8.64e-4` 降到 `6.88e-4 eV/muB`
  （改善 20.4%）；普通切片的能量、磁力中位数改善 3.8%/3.2%，原子力
  退化 4.1%。因此它补的是实测响应能力，不宣称所有聚合指标单调变好。
- 保留 `correlation_same_edge`。它和 `edge_l0_dot2` 属于同一
  两体拓扑，但在代表性采样中仍贡献约 2.5%–6% 的独立径向信息。对
  初始化径向族做一次 Gram–Schmidt 会在径向系数训练后立即失效；
  每步动态正交化则增加全局归约、导数和格式语义，不值得为这点
  病态性付出。
- 不加入六组 density cross-rank 通道。在 generic/bcc-random 采样中，
  它们相对现有通道的线性残差仅 1.05%/0.22%；即使把目标刻意设为该
  缺失通道本身，保零 scaler 下的三种子 MLP 中位改进也只有约 10%，
  而生产维数会增加约 11%，还要增加六个中心收缩。目前没有真实关键
  切片收益足以支付这组持续的推理与导数成本。
- warm-up 不增加 O2-only 描述符分支。最终共享 CUDA 路径的完整训练
  开销仅比 spin2 高 3.74%，为省一段 warm-up 的小比例计算引入第二套
  布局和图路径会扩大长期维护面。

`nep4_spin3` 文本模型需要识别该 header 和 55 维排布的下游 reader；
只支持 `nep4_spin2` 的 GPUMD/NEPAdapters reader 不能直接加载它。
