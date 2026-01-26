***************************************************************
* 倾向得分诊断验证脚本
* 用于生成 Python 测试的 Stata 参考值
* 
* Reference: Story 1.1 - 倾向得分诊断增强
***************************************************************

clear all
set more off

* 设置工作目录 - 根据实际路径修改
cd "/Users/cxy/Desktop/大样本lwdid/Lee_Wooldridge_2023-main 3"

* 加载数据
use "2.lee_wooldridge_staggered_data.dta", clear

***************************************************************
* Cohort 4, Period 5 诊断
***************************************************************
preserve

di _n "=========================================="
di "   Cohort 4, Period 5 倾向得分诊断"
di "=========================================="

* 子样本选择: cohort 4 + not-yet-treated at period 5
* 注意: 数据使用 g0 表示 never-treated
keep if (g4 == 1) | (g6 == 1) | (g0 == 1)
keep if f05 == 1

* 样本量检查
count
di "总样本量: " r(N)
count if g4 == 1
di "处理组 (cohort 4): " r(N)
count if g4 == 0
di "控制组 (cohort 6 + inf): " r(N)

* 估计倾向得分（原始，无裁剪）
logit g4 x1 x2
predict ps_raw, pr

* 基础统计（裁剪前）
di _n "=== 裁剪前倾向得分统计 ==="
summarize ps_raw, detail
scalar ps_raw_mean = r(mean)
scalar ps_raw_sd = r(sd)
scalar ps_raw_min = r(min)
scalar ps_raw_max = r(max)
scalar ps_raw_p25 = r(p25)
scalar ps_raw_p50 = r(p50)
scalar ps_raw_p75 = r(p75)

di "Mean: " ps_raw_mean
di "SD: " ps_raw_sd
di "Range: [" ps_raw_min ", " ps_raw_max "]"
di "Quantiles: [" ps_raw_p25 ", " ps_raw_p50 ", " ps_raw_p75 "]"

* 裁剪倾向得分
gen ps_trim = ps_raw
replace ps_trim = 0.01 if ps_trim < 0.01
replace ps_trim = 0.99 if ps_trim > 0.99

* 基础统计（裁剪后）
di _n "=== 裁剪后倾向得分统计 ==="
summarize ps_trim, detail
scalar ps_trim_mean = r(mean)
scalar ps_trim_sd = r(sd)
scalar ps_trim_min = r(min)
scalar ps_trim_max = r(max)
scalar ps_trim_p25 = r(p25)
scalar ps_trim_p50 = r(p50)
scalar ps_trim_p75 = r(p75)

di "Mean: " ps_trim_mean
di "SD: " ps_trim_sd
di "Range: [" ps_trim_min ", " ps_trim_max "]"
di "Quantiles: [" ps_trim_p25 ", " ps_trim_p50 ", " ps_trim_p75 "]"

* 权重 CV（仅控制组）
di _n "=== 控制组权重统计 ==="
gen weight = ps_trim / (1 - ps_trim) if g4 == 0

summarize weight
scalar weight_mean = r(mean)
scalar weight_sd = r(sd)
scalar weight_cv = weight_sd / weight_mean

di "Weight mean: " weight_mean
di "Weight SD: " weight_sd
di "Weight CV: " weight_cv

if weight_cv > 2.0 {
    di "⚠️ 警告: 权重CV过高 (CV=" weight_cv " > 2.0)"
}

* 极端值检测（裁剪前）
di _n "=== 极端值检测 ==="
count if ps_raw < 0.01
scalar n_extreme_low = r(N)
count if ps_raw > 0.99
scalar n_extreme_high = r(N)
count
scalar n_total = r(N)

scalar extreme_low_pct = n_extreme_low / n_total
scalar extreme_high_pct = n_extreme_high / n_total
scalar extreme_total_pct = extreme_low_pct + extreme_high_pct
scalar n_trimmed = n_extreme_low + n_extreme_high

di "Low extreme (ps < 0.01): " n_extreme_low " (" extreme_low_pct*100 "%)"
di "High extreme (ps > 0.99): " n_extreme_high " (" extreme_high_pct*100 "%)"
di "Total extreme: " n_trimmed " (" extreme_total_pct*100 "%)"

if extreme_total_pct > 0.10 {
    di "⚠️ 警告: 极端值比例过高 (" extreme_total_pct*100 "% > 10%)"
}

* 输出完整诊断报告
di _n "========================================="
di "        Python 对照参考值 (g=4, r=5)"
di "========================================="
di "STATA_REFERENCE_G4_P5 = {"
di "    'ps_mean_trimmed': " ps_trim_mean ","
di "    'ps_std_trimmed': " ps_trim_sd ","
di "    'ps_min_trimmed': " ps_trim_min ","
di "    'ps_max_trimmed': " ps_trim_max ","
di "    'ps_p25': " ps_trim_p25 ","
di "    'ps_p50': " ps_trim_p50 ","
di "    'ps_p75': " ps_trim_p75 ","
di "    'weights_cv': " weight_cv ","
di "    'extreme_low_pct': " extreme_low_pct ","
di "    'extreme_high_pct': " extreme_high_pct ","
di "    'n_trimmed': " n_trimmed ","
di "    'n_total': " n_total ","
di "}"
di "========================================="

restore

***************************************************************
* 其他 Cohort/Period 组合 (可选)
***************************************************************

* 可以添加更多组合的验证...

di _n "验证脚本执行完毕"
