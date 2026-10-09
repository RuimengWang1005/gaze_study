# Equivalence verification

比较对象：`global_prechange`、`global_postchange`、`large_radius_local`；样本按 `image_id` 对齐。浮点判定容差为 `atol=1e-6, rtol=1e-6`。`records.jsonl` 实际存在；VIS scores 对应 `tokens/*.npz:raw_score`，Focal/Spatial/final indices 对应 records 中的 `focal_indices` / `spatial_indices` / `selected_indices`。

## 配置核对

三次运行 checkpoint 均为 `/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B`，`config_sha256=a987f1aa38e5f0a411e262a8d3739aaafd11d34f4ddc484cdeef4f68e2685d86`；checkpoint 元数据稳定 SHA-256 均为 `05738112c7eade07e5fcc2bfae5185b50540aa6904e8d665a91390da7a0534a4`（这是配置及权重文件名/大小/mtime 的 hash，不是权重内容 hash）。

| 配置 | global_prechange | global_postchange | large_radius_local |
|---|---:|---:|---:|
| seed / 样本数 | 42 / 5 | 42 / 5 | 42 / 5 |
| budget / 实际 retain ratio | 0.20 / 0.1978021978–0.20 | 同左 | 同左 |
| redundancy mode | Global（显式字段 `NOT RECORDED`；旧实现路径） | Global（显式字段 `NOT RECORDED`；默认路径） | Local |
| local radius | `NOT RECORDED`（Global 下不适用） | `NOT RECORDED`（Global 下不适用） | 64 |
| final / Focal / Spatial token 数 | 53–56 / 48–50 / 5–6 | 同左 | 同左 |
| Spatial / Focal 预算规则 | retained K 的 0.10 / 其余 0.90 | 同左 | 同左 |
| cosine threshold / penalty strength | 0.7 / 1.0 | 0.7 / 1.0 | 0.7 / 1.0 |
| IQR / alpha | 独立 IQR 配置 `NOT RECORDED`；alpha/weight=1.0 | 同左 | 独立 IQR 配置 `NOT RECORDED`；诊断 `adaptive_alpha=1.0` |
| prompt | `prompt_query=null`（数据集默认）；prompt bundle SHA-256 `c6b6ec4d7722e320809cf06eb81ef3f17a0bd6eb9bc6628ad24864c37b4e3590` | 同左 | 同左 |
| decoding | temperature=0.0, top_p=null, beams=1, max_new_tokens=32, choices=ABCD | 同左 | 同左 |

逐字段核对时忽略了允许差异（run ID、输出目录、运行耗时/显存统计、prechange↔postchange 代码版本，以及 large-radius 的 mode/radius）。其余非预期差异：

| 比较 | 字段 | 差异 |
|---|---|---|
| postchange vs large-radius | `source_hashes.audit_fgvc_selection.py` | `cd7dd89…` vs `957d75b…` |
| postchange vs large-radius | `source_hashes.token_compression/selector_model.py` | `05137321…` vs `ccf5f097…` |
| postchange vs large-radius | `git_status` | 工作树状态及未跟踪文件集合不同 |
| postchange vs large-radius | `audit_args.timing` | 字段缺失 vs `false`（虽为关闭状态，仍是 schema 差异） |

因此严格配置一致性为 **FAIL**；除上述代码/记录 schema 差异及明确允许项外，未发现其他配置差异。

## Manifest 核对

| 项目 | 结果 | 证据 |
|---|---|---|
| sample ID 集合 | PASS | 三者均为 `{1,10,19,22,48}` |
| manifest / records 顺序 | PASS | 三者均为 `[22,10,19,48,1]` |
| ground truth | PASS | `class_id`、records 的 `gold_label` 均逐样本一致；bundle SHA-256 `68fc2ceb12b1612e6423b0ae0f0589d35115cddafb6f50516f327d152a5805af` |
| 图像路径/标识 | PASS | `image_id` 与 `relative_path` 逐项一致 |
| annotation hashes | PASS | 六项逐字段一致 |

三个 `manifest.json` 的原始文件 SHA-256 均为 `a394830b89ebb5c1c15a625a45baf84c5c8e1fa7dd853f291b57c06db7fe3bec`；canonical JSON SHA-256 均为 `4e5f6a832ef2c4347f0b20771e0abc6623035b5a79dd4aedf6b5a520226d8f91`。

**MANIFEST CONSISTENCY: PASS**

## 修改前 Global vs 修改后 Global

| 指标 | 结果 |
|---|---:|
| 所有已记录目标字段完全相同样本 | 5/5 |
| 第一个不一致 sample ID / 阶段 | 无 / 无 |
| VIS scores 最大绝对误差 | 0 |
| 其他已记录浮点数组最大绝对误差 | 0（含 token raw scores、`max_selected_local_cosine`） |
| Focal / Spatial / final indices 完全一致率 | 100% / 100% / 100% |
| Focal / Spatial / final token 数一致率 | 100% / 100% / 100% |
| prediction raw / parsed 一致率 | 100% / 100% |
| adjusted scores / max similarity / penalty | `NOT RECORDED`（两次 Global records 均无这些数组） |

所有可比较浮点量均逐元素精确相等，因而也满足数值容差。

**GLOBAL REGRESSION: PASS**

## 修改后 Global vs 大半径 Local

| 指标 | 结果 |
|---|---:|
| 所有已记录且可比较目标字段完全相同样本 | 5/5 |
| 不一致 sample ID / 第一个不一致阶段 | 无 / 无 |
| VIS scores 最大绝对误差 | 0 |
| 其他可比较浮点数组最大绝对误差 | 0（含 token raw scores、`max_selected_local_cosine`） |
| Focal / Spatial / final indices 完全一致率 | 100% / 100% / 100% |
| Focal / Spatial / final token 数一致率 | 100% / 100% / 100% |
| prediction raw / parsed 一致率 | 100% / 100% |
| max similarity / penalty base | Local 已记录；Global `NOT RECORDED`，不能直接逐数组比较 |
| adjusted score | `NOT RECORDED`（两次运行均未保存） |

| image_id | merger grid (Hm×Wm) | 覆盖整图所需最大半径 | radius=64 覆盖 | outside-local ratio / empty-neighbor ratio |
|---:|---:|---:|:---:|---:|
| 22 | 13×21 | 20 | PASS | 0 / 0 |
| 10 | 15×18 | 17 | PASS | 0 / 0 |
| 19 | 14×19 | 18 | PASS | 0 / 0 |
| 48 | 20×14 | 19 | PASS | 0 / 0 |
| 1 | 14×20 | 19 | PASS | 0 / 0 |

每次输入只有一个图像 group；半径 64 覆盖每张图的完整 merger 后 token grid。可观察选择结果全部精确等价，且诊断显示不存在 group mask 越界、radius 不足或空邻域现象。由于 Global 未保存内部 max-similarity/penalty，内部数组本身不能做直接数值复核。

**LARGE-RADIUS EQUIVALENCE: PASS**

## 最终结论

| CheckResultKey | Result | Evidence |
|---|---|---|
| Config consistency | FAIL | postchange 与 large-radius 的两个 source hash、worktree 状态及 `timing` schema 存在非预期差异 |
| Manifest consistency | PASS | 三份原始及 canonical manifest hash 分别完全一致 |
| Global regression | PASS | 5/5；indices/token 数/prediction 全同；最大浮点误差 0 |
| Large-radius equivalence | PASS | 5/5；可比较输出全同；radius 64 完整覆盖全部网格 |
