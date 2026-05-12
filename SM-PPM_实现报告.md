# SM-PPM 实现报告

## 1. 结论先说

你给出的这段描述：

> SM-PPM 本质上是 source-dependent TTA，会使用源域信息，但通常不是“测试时每一步都反复读取原始 source image”；更常见的是测试前先提取 prototype / memory / prior，测试时主要访问源域表征。

**对这份代码库里的 `smppm` 来说，这个说法只在特定变体里部分成立；对默认实现并不成立。**

- 对 `TCA/segmentation` 里的原版 `SM-PPM`：
  - **不成立**
  - 它在测试时的每个 target batch / adaptation step，都会从 `src_loader` 里取一批 **source image + source label** 来更新模型。
  - 它不是“测试前缓存 source prototype，测试时不再读 source image”的实现。

- 对 `DCON` 里的 `SM-PPM` 默认模式 `smppm_ablation_mode=full`：
  - **也不成立**
  - 它同样在测试时持续读取 **labeled source batch**。
  - 只是它比原版多了一层：先从 target 提取 prototype，再用 target 的统计量对 source image 做 AdaIN 风格混合，然后仍然用 source label 做监督更新。

- 对 `DCON` 里的 `smppm_ablation_mode=source_free_proto`：
  - **才接近你说的“只依赖 target / prototype，不读 source image”**
  - 但这已经不是默认 `SM-PPM full`，而是你们代码里新增的 source-free ablation。

所以，**如果你问“我们代码里的 smppm 默认是不是只用 source prototype / source prior，而不是持续读取 source image？”答案是否定的。默认实现是显式 source-dependent，而且测试时在线读取 source batch。**

---

## 2. 代码位置

代码库里有两套 `smppm`：

1. `TCA/segmentation`
   - 方法实现：`/Users/RexRyder/PycharmProjects/TTA/TCA/segmentation/methods/sm_ppm.py`
   - 测试入口：`/Users/RexRyder/PycharmProjects/TTA/TCA/segmentation/test_time.py`

2. `DCON`
   - 方法实现：`/Users/RexRyder/PycharmProjects/TTA/DCON/models/tta_smppm.py`
   - 训练器接入：`/Users/RexRyder/PycharmProjects/TTA/DCON/models/exp_trainer.py`
   - 参数与数据加载：`/Users/RexRyder/PycharmProjects/TTA/DCON/train.py`

---

## 3. TCA 原版 SM-PPM 是怎么工作的

### 3.1 初始化就要求 source loader

`SMPPM` 构造函数直接接收 `src_loader`，并保存 `self.src_loader_iter`：

- `TCA/segmentation/methods/sm_ppm.py:19-23`

测试入口 `setup_smppm()` 也明确把 `setup_src_loader(cfg, IMG_MEAN)` 传进去：

- `TCA/segmentation/test_time.py:188-197`

这说明它不是 source-free，也不是只加载一个离线 prototype 文件。

### 3.2 每次测试时都会读一批 source 数据

核心逻辑在 `forward_and_adapt()`：

1. 先对当前 target 图像前向，提取 target feature。
   - `sm_ppm.py:47-52`

2. 然后从 `src_loader_iter` 里取一批 source：
   - `sm_ppm.py:54-59`

3. 用 source image 前向，结合 target feature 做 SM 风格混合：
   - `sm_ppm.py:61`

4. 再用 source label 计算损失并更新：
   - `sm_ppm.py:71-73`

也就是说，**target 只是提供 feature/prototype 参考，真正驱动参数更新的监督信号仍来自在线读取的 source label。**

### 3.3 它用的不是“预存 source prototype”

这版的 prototype 实际来自 **target feature patch**：

- `sm_ppm.py:47-52`

然后用 source feature 和这些 target patch prototype 做相似度：

- `sm_ppm.py:65-69`

最终权重是：

- `confidence * (1 - entropy_source)`
  - `sm_ppm.py:71`

所以这版并不是：

- 测试前先抽取 source prototype
- 测试时 target 去对齐 source prototype

而是：

- 测试时先从 target 提取 patch prototype
- 再拿一批 source 做监督更新

---

## 4. DCON 版 SM-PPM 是怎么工作的

### 4.1 默认模式仍然需要 source loader

`DCON/train.py` 里明确规定：

- 只要 `opt.tta == 'sm_ppm'` 且 `smppm_ablation_mode != 'source_free_proto'`
- 就认为 **需要 source loader**

见：

- `DCON/train.py:1468-1477`

并且日志直接写明：

- “SM-PPM source loader: using labeled source-domain training split...”
- `DCON/train.py:1565-1573`

参数定义里也写得很直白：

- `--smppm_steps`：每个 target batch 使用多少个 labeled source batch
  - `DCON/train.py:392-393`
- `--smppm_src_batch_size`：source-domain labeled batch size
  - `DCON/train.py:394-395`

这已经足够说明默认 `DCON` 版不是“只访问源域表征”。

### 4.2 适配器初始化要求 source_loader

`configure_smppm()` 里，如果不是 `source_free_proto`，没有 `source_loader` 就直接报错：

- `DCON/models/exp_trainer.py:799-807`

随后把这个 loader 传给 `SMPPMAdapter`：

- `DCON/models/exp_trainer.py:821-838`

### 4.3 默认 full 模式的在线流程

`SMPPMAdapter` 注释已经把流程写出来了：

- 先提 target bottleneck feature 并做 patch prototype
- 再取一批 labeled source
- 再用 source label 做更新
- 最后再预测 target

见：

- `DCON/models/tta_smppm.py:72-81`

具体实现如下。

#### Step A: 从 target 提取 prototype

- `DCON/models/tta_smppm.py:298-313`

这里 prototype 来自 **target bottleneck feature patch pooling**，不是 source prototype bank。

#### Step B: 每一步适配都取 source batch

- `DCON/models/tta_smppm.py:261-271`
- `DCON/models/tta_smppm.py:487-497`

这两段非常关键，说明 **每个 target batch 的每个 adaptation step 都会 `next(self.source_iter)`**。

#### Step C: 可选 SM

如果 ablation mode 使用 SM（`full` 或 `sm_ce`），会把 target 图像统计量迁移到 source image：

- `DCON/models/tta_smppm.py:112-116`
- `DCON/models/tta_smppm.py:347-356`

具体实现是 tensor-space AdaIN：

- `DCON/models/tta_smppm.py:46-56`

注意这里仍然是 **source image 参与前向**，只是 source image 被 target statistics 风格化了。

#### Step D: 可选 PPM

如果 ablation mode 使用 PPM（`full` 或 `ppm_ce`），会计算：

- source feature 和 target patch prototype 的相似度
  - `DCON/models/tta_smppm.py:315-321`
- source prediction entropy
  - `DCON/models/tta_smppm.py:323-328`
- pixel weight = confidence * (1 - entropy)
  - `DCON/models/tta_smppm.py:362-375`

#### Step E: 用 source label 做监督更新

- CE-only ablation：`DCON/models/tta_smppm.py:377-378`
- 默认 full：`DCON/models/tta_smppm.py:379-380`

默认 full 用的是 DCON 自己的加权 Dice + CE：

- `DCON/models/exp_trainer.py:723-748`

### 4.4 target label 不参与适配

`te_func_smppm()` 里注释写得很清楚：

- target label 只用于 evaluation
- adaptation 使用 source label 和 target image feature

见：

- `DCON/models/exp_trainer.py:1685-1705`

---

## 5. 你那段描述和代码实现的对应关系

### 5.1 哪些点是对的

你的描述里这些点，**对 `source_free_proto` 变体，或者对一些论文里的 source-dependent TTA 泛化描述，是成立的**：

- “source-dependent TTA 会使用源域信息”
- “测试时可以通过 prototype / memory / prior 约束 target adaptation”
- “不一定每步重新读取原始 source image”

### 5.2 哪些点对这份代码默认实现不对

如果指的是这份仓库里默认跑的 `smppm`：

- `TCA` 原版 `sm_ppm`
- `DCON` 默认 `smppm_ablation_mode=full`

那么下面这句 **不对**：

> 主要访问的是“源域表征”，不是 dataloader 那种持续读 source image

因为实际代码就是：

- 有 `source_loader`
- 每个 target batch / adaptation step 都 `next(source_iter)`
- 读取 source image 和 source label
- 用 source label 做在线监督更新

### 5.3 更准确的表述

更准确可以写成：

> 这份代码库里的默认 `SM-PPM` 是一种 **source-dependent、online supervised TTA**。  
> 它在测试时不仅使用 source-domain information，而且会持续从 `source_loader` 读取 **source image + source label**。  
> target 侧主要提供 patch prototype 或风格统计量；真正的更新监督仍来自 source batch。  
> 只有 `DCON` 中新增的 `source_free_proto` 变体，才是不读取 source batch 的 target-only 近似 source-free 版本。

---

## 6. 各模式归类

### TCA 原版

- `sm_ppm`
  - 类型：source-dependent
  - 测试时是否读 source image：**是**
  - 测试时是否用 source label：**是**
  - 是否只依赖 source prototype / prior：**否**

### DCON 扩展版

- `smppm_ablation_mode=full`
  - 类型：source-dependent
  - 测试时是否读 source image：**是**
  - 测试时是否用 source label：**是**
  - target 提供：prototype + style statistics

- `smppm_ablation_mode=source_ce_only`
  - 类型：source-dependent
  - 测试时是否读 source image：**是**
  - 测试时是否用 source label：**是**
  - 不用 SM / PPM，仅 source CE

- `smppm_ablation_mode=sm_ce`
  - 类型：source-dependent
  - 测试时是否读 source image：**是**
  - 测试时是否用 source label：**是**
  - 用 target statistics 做 source style mixing

- `smppm_ablation_mode=ppm_ce`
  - 类型：source-dependent
  - 测试时是否读 source image：**是**
  - 测试时是否用 source label：**是**
  - 用 target prototype 给 source pixel weighting

- `smppm_ablation_mode=source_free_proto`
  - 类型：target-only / source-free ablation
  - 测试时是否读 source image：**否**
  - 测试时是否用 source label：**否**
  - 使用：target pseudo-label confidence、masked entropy、target prototype compactness
  - 代码：`DCON/models/tta_smppm.py:391-475, 570-663`

---

## 7. 最终判断

如果你现在要确认“`smppm` 对应的方法是不是下面这段描述”，我的最终判断是：

- **不是完全对应。**
- **对默认实现来说，结论相反一半。**

更具体地说：

1. `smppm` 在这份代码里确实是 **source-dependent TTA**。
2. 但默认实现 **不是** “只访问 source prototype / source prior，不持续读 source image”。
3. 默认实现是 **测试时在线读取 source batch，并用 source label 做监督更新**。
4. 只有 `DCON` 里新增的 `source_free_proto` 变体，才接近你描述的那种“部署后隔离 source image”的形式。

