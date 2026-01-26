* PSM 验证测试 - 横截面数据
clear all
import delimited "tests/data/psm_crosssection_small.csv", clear

* 运行 PSM
teffects psmatch (y) (d x1 x2), atet nn(1) vce(robust)

* 输出结果
matrix b = e(b)
matrix V = e(V)
di "=== Stata PSM Results ==="
di "ATT: " b[1,1]
di "SE:  " sqrt(V[1,1])
di "n_treat: " e(n1)
di "n_control: " e(n0)
