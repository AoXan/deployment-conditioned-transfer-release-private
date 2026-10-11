# 实际执行与测试说明

安装requirements-verified.lock，并按DATA_ACQUISITION_GUIDE.md合法获取真实数据。公开CY-Bench代表性训练：

```sh
python scripts/acquire_cybench_subset.py --output-root data/external --pair wheat_AU
python scripts/run_real_data_smoke.py --raw-root data/external/cybench/raw --output outputs/main_real_01
python scripts/render_new_experiment_assets.py --source outputs/main_real_01 --output outputs/main_real_01/tables
```

真实AU原始六表→build_cybench_view预处理→原regional split→execute_job执行监督/预测KD/表征KD/联合KD/reference训练→新checkpoint/preprocessor/predictions→SHAP/IG/统计→新表图。脚本支持--component选择训练分支。主US→AU路径由 `run_primary_source_target_smoke.py --config CONFIG --output NEW` 重新训练源模型、screen/qualify，再训练GROUP目标；不是读取历史Checkpoint。

原Stage7/8、Roseworthy、Waite、G2F、授权原始清洗数据及APSIM路径见EXPERIMENT_EXECUTION_GUIDE.md。已实际测试490真实region-year、5训练组件、SHAP/IG、US→AU迁移及11实际APSIM场景。不可获得的授权输入/历史Checkpoint bundle仍为条件，不把synthetic fixture或冻结CSV校验当作农业科研复现。

`make smoke-real`执行真实公开数据训练。`make reproduce-full`执行授权预处理和formal训练，需要原plan fingerprint。`make reproduce-results`与旧 `make smoke` 是冻结输出/fixture验证，不能当作训练成功证据；这一点明确保留标签。

接受稿与原科学结论原样保留。main整合保留原main的发布整理和accepted预处理快照，修复入口另接真实执行。状态CONDITIONALLY_REPRODUCIBLE，不保证所有授权数据/资源受限实验的全量数值。
