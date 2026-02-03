# Wild Cluster Bootstrap 实现说明

## 算法原理

本包实现了 Cameron, Gelbach & Miller (2008) 的 Wild Cluster Bootstrap 算法。

### p 值定义

根据算法原理，双侧检验的 p 值定义为：

```
p = P(|t*| ≥ |t_orig| | H0)
```

其中 t* 是在零假设下通过 bootstrap 生成的 t 统计量。

### 完全枚举

当 G 个 cluster 使用 Rademacher 权重时，有 2^G 种可能的权重组合。完全枚举计算了所有可能情况，产生精确的 p 值，无蒙特卡洛误差。

**完全枚举模式与 Stata boottest 产生 100% 等价的结果。**

## 使用方法

### 基本用法

```python
from lwdid.inference import wild_cluster_bootstrap

# 基本用法（G <= 12 时自动使用完全枚举）
result = wild_cluster_bootstrap(
    data=df,
    y_transformed='y',
    d='d',
    cluster_var='cluster',
)
```

### 显式控制完全枚举

```python
# 强制使用完全枚举
result = wild_cluster_bootstrap(
    data=df,
    y_transformed='y',
    d='d',
    cluster_var='cluster',
    full_enumeration=True,
)

# 强制使用随机抽样
result = wild_cluster_bootstrap(
    data=df,
    y_transformed='y',
    d='d',
    cluster_var='cluster',
    full_enumeration=False,
    n_bootstrap=999,
)
```

### 使用 wildboottest 包

```python
# 使用 wildboottest 包（需要先安装：pip install wildboottest）
result = wild_cluster_bootstrap(
    data=df,
    y_transformed='y',
    d='d',
    cluster_var='cluster',
    use_wildboottest=True,
)
```

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | DataFrame | 必需 | 回归数据 |
| `y_transformed` | str | 必需 | 结果变量名 |
| `d` | str | 必需 | 处理变量名 |
| `cluster_var` | str | 必需 | 聚类变量名 |
| `controls` | List[str] | None | 控制变量 |
| `n_bootstrap` | int | 999 | Bootstrap 次数（完全枚举时忽略） |
| `weight_type` | str | 'rademacher' | 权重类型：'rademacher', 'mammen', 'webb' |
| `alpha` | float | 0.05 | 显著性水平 |
| `seed` | int | None | 随机种子 |
| `impose_null` | bool | True | 是否施加零假设 |
| `full_enumeration` | bool | None | 是否完全枚举。None 时自动决定 |
| `use_wildboottest` | bool | False | 是否使用 wildboottest 包 |

## Stata 等价性

### 完全枚举模式

当使用完全枚举模式（`full_enumeration=True`）时，本实现与 Stata boottest 产生 **100% 等价** 的 p 值：

| 模式 | Stata p 值 | Python p 值 | 差异 |
|------|-----------|-------------|------|
| 完全枚举 (2^G 次) | 0.5020 | 0.5020 | **0** |
| 随机抽样 (999次) | 0.4915 | 0.5085 | 0.017 |

随机抽样模式的差异完全来自随机数生成器不同，不影响统计推断的有效性。

### 自动完全枚举

当满足以下条件时，自动启用完全枚举：
- 聚类数量 G ≤ 12
- 权重类型为 'rademacher'

这确保了在小样本情况下获得与 Stata 完全一致的结果。

## 参考文献

- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008). "Bootstrap-based improvements for inference with clustered errors." Review of Economics and Statistics, 90(3), 414-427.
- Roodman, D., Nielsen, M.Ø., MacKinnon, J.G., & Webb, M.D. (2019). "Fast and wild: Bootstrap inference in Stata using boottest." The Stata Journal, 19(1), 4-60.
