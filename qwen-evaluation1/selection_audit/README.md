# CUB 选点审计（第一阶段）

这是只读诊断工具。复用 `eval_fgvc_selector.py` 的加载、processor、prompt 和生成逻辑，
不改模型权重、scorer、余弦惩罚或选点。`selector_model.py` 无修改。

**CUB part coverage 只是解剖覆盖代理，不等于真实类别判别 utility。**

## 基线与运行

从已有结果 JSON 的 `config` 读取全部模型参数，缺失关键参数直接报错。
默认参考 `20260902_153802_results.json`：checkpoint 为
`/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B`，CUB 为
`/data/DataSet/vlm_dataset/CUB_200_2011`，options 位于 `/home/ruimeng/FinePruner_TIP2026/data/`。
脚本与 Python 默认的 cosine weight=1、threshold=0.7、有效 spatial ratio=0.1、seed=42 一致；
脚本循环 budget=0.1/0.2，Python 默认 0.1。脚本像素范围 200704–1605632，
Python 默认 min=max=3686400，**不能直接采用 Python 默认像素设置**。
selector 类自身环境回退的 weight=0.15、threshold=0.85 也不会用于本次审计。

```bash
cd /home/ruimeng/gaze/qwen-evaluation1
AUDIT_TAG=$(date -u +%Y%m%dT%H%M%SZ)
bash run_selection_audit.sh --mode fgvc-options --budget 0.1 --limit 5 --equivalence --run-id "eq5_${AUDIT_TAG}"
bash run_selection_audit.sh --mode cub-train --budget 0.1 --limit 20 --run-id "train20_b01_${AUDIT_TAG}"
bash run_selection_audit.sh --mode cub-train --budget 0.1 --run-id "train200_b01_${AUDIT_TAG}"
bash run_selection_audit.sh --mode cub-train --budget 0.2 --manifest "audit_outputs/train200_b01_${AUDIT_TAG}/manifest.json" --run-id "train200_b02_${AUDIT_TAG}"
```

所有输出在新建的 `audit_outputs/<run_id>/`。已有 run-id 会报错，不覆盖。
可用 `BASELINE_RESULT` / `PYTHON_BIN` / `CUDA_VISIBLE_DEVICES` 指向已确认的环境。
默认 GPU=3、vs_env、离线运行。除 budget 和样本数量外不覆盖基线模型参数。
train 默认 seed 来自基线（42），每类随机一张训练图，再随机排列类别；20 张为同一列表前 20 张。
复用 manifest 会校验 seed、标注 SHA256、训练 split。options 模式保留原评测样本和选项顺序，
只用于等价检查或错误分析，不用于调参。train 使用固定短 prompt 做一次完整 forward，不生成答案。

## 坐标与不变量

已检查实际安装的 `qwen_vl_utils.fetch_image` 和 `Qwen2VLImageProcessor._preprocess`：
整图 resize，无裁剪、空间 padding 或 letterbox。静态图片复制 temporal patch 不改变空间映射。
记录实现文件及 SHA256；其他 processor、已开启裁剪/padding 或旋转 EXIF 会拒绝运行。
视觉 encoder 的 window 顺序在 merger 后通过 reverse_indices 恢复，因此索引为行优先。
`t=1`；`Hm=grid_h/merge_size`；`Wm=grid_w/merge_size`；`N=Hm*Wm`。
token `(row,col)=(i//Wm,i%Wm)`；part 使用 `floor(y/H*Hm),floor(x/W*Wm)` 并 clamp。
依照任务定义直接使用 CUB 原坐标，不额外做减一变换。

每张图断言索引范围、分支无重复、union=selected、K 和分支预算、scores/features 无 NaN/Inf。
用相同 scorer 对缓存 features 重算 raw scores，和 cached selected scores 比较
`rtol=1e-4, atol=1e-5`。不读取 `last_combined_scores` 作为 raw score。
原评测等价检查关闭缓存，只用返回值 hook 记录选点；不替换 forward，不修改其返回值。
与开启缓存的审计比较完整预测、选项、selected indices、N、K 和全部模型配置。

## 指标口径

- 空间集中：8 邻接分量、最大分量比例、3×3/5×5 最大密度、存在相邻点的 token 比例、
  4×4 归一化宏网格覆盖。focal 以 K_focal 为分母，union 以 K 为分母。
- 局部重复：Chebyshev 距离≤1，余弦≥0.7；距离≤2 独立做敏感性分析。
  阈值仅用于诊断，可显式配置并被记录；不影响选点。
  LocalDuplicateRate 是至少有一个相似邻居的 token 比例；ExcessDuplicateRatio 是
  各连通簇除一个代表外的成员数/分支 token 数；LargestLocalClusterShare 仅统计大小≥2 的簇。
- coverage 仅用 visible=1；分别记录 exact-cell 和 Chebyshev 1-cell。
  SpatialRescue 分母为全部可见部件；Conditional 版本分母为 focal 漏失部件。
  summary 同时保存可见部件 micro 平均和每图平均。不存在可见部件的 recall 为 null。
- 五类宏观部件按任务指定分组。macro any-part recall 是可见标注点至少一个命中；
  另提供宏观组内全部可见标注点的 part recall。
- 漏失归因使用 union 的 1-cell miss，在 3×3 邻域找最高 raw-score token。
  rank 是 1-based，分数并列按原 token index 升序；raw Top-K_focal 使用相同排序。
  rank>2K_focal 为 scorer_miss，rank≤K_focal 且未选为 selection_suppression，其他为 near_candidate。
  这是观测分类，不证明模型真实 utility 或因果关系。
- ActionableRedundancyCase 要求整个可见宏观部件漏失、其邻域有 rank≤2K_focal 候选，
  且 focal 存在 radius=1 的局部重复簇多余成员。
- bbox 外比例按 token 中心点是否在 bbox 内计算。窗口统计在图像边缘以零扩展。

## 输出和可视化

run_config 记录基线来源、完整参数、checkpoint 配置 hash 和权重文件大小/mtime、
代码/预处理 hash、commit、工作树状态、GPU 峰值内存和耗时。权重没有重写或完整哈希。
records.jsonl 每图引用 run_config，包含分支索引/坐标/raw score、部件命中、排名和归因。
NPZ 不保存 D 维 features；selection_type=0 未选、1 focal、2 spatial；
max_selected_local_cosine=-2 表示没有选中的局部邻居。临时 features 在每图统计后释放。
per_sample/per_part/summary 同时提供 CSV，完整嵌套结构以 JSON 为准。

overlays/selection.json 保存独立 seed 随机 20 张与排名前 20 张的名单（合并去重保存图片）。
排名只用于查看：符合 actionable 条件时，候选漏失宏观部件数×focal 多余成员数；否则为 0。
这不是新的研究综合指标。若 actionable 不足 20 张，余位按 image_id 填入。
四面板包含原图、部件、raw 热图、raw Top-K_focal、橙色 focal、蓝色 spatial，
绿色/紫色表示 1-cell 覆盖/漏失，红色虚线为最大重复簇外接框。
坐标需要结合随机叠加图人工检查，自动断言不能证明语义部件标注正确。

## CPU 测试

```bash
python -m compileall -q audit_fgvc_selection.py selection_audit tests/test_selection_audit.py
python -m pytest -q tests/test_selection_audit.py
git diff --check
```

无需加载模型或 GPU。若环境没有 pytest，可只安装到临时目录，避免改变复现环境。
