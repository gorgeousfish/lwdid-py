"""测试common timing和staggered模块的HC标准误实现"""
import numpy as np
import pandas as pd

print("=" * 60)
print("测试 HC 标准误实现")
print("=" * 60)

# ============================================================
# 测试 1: staggered/estimation.py
# ============================================================
print("\n" + "=" * 60)
print("1. 测试 staggered/estimation.py")
print("=" * 60)

from lwdid.staggered.estimation import (
    run_ols_regression,
    _compute_leverage,
    _compute_hc_variance,
    _compute_hc1_variance,
)

# 生成测试数据
np.random.seed(42)
n = 100
x1 = np.random.randn(n)
d = np.random.binomial(1, 0.5, n)
y = 1 + 2*d + 0.5*x1 + np.random.randn(n) * np.sqrt(1 + x1**2)

data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})

# 测试各种 vce 类型
vce_types = [None, 'hc0', 'hc1', 'robust', 'hc2', 'hc3', 'hc4']

print("\n测试 run_ols_regression 各 vce 类型:")
print("-" * 60)
results_staggered = {}
for vce in vce_types:
    try:
        result = run_ols_regression(data, 'y', 'd', vce=vce)
        results_staggered[vce] = result
        print(f"vce='{vce}': ✅ ATT={result['att']:.4f}, SE={result['se']:.4f}")
    except Exception as e:
        print(f"vce='{vce}': ❌ {type(e).__name__}: {e}")

# ============================================================
# 测试 2: lwdid() API (common timing)
# ============================================================
print("\n" + "=" * 60)
print("2. 测试 lwdid() API (common timing)")
print("=" * 60)

from lwdid import lwdid

# 生成common timing面板数据
np.random.seed(123)
n_units = 50
n_periods = 6
treatment_start = 4

ids = []
times = []
ys = []
ds = []
posts = []

# 单位固定效应
unit_fe = np.random.randn(n_units) * 2

# 处理分配
treated = np.random.binomial(1, 0.5, n_units)

for i in range(n_units):
    for t in range(1, n_periods + 1):
        ids.append(i)
        times.append(t)
        post = 1 if t >= treatment_start else 0
        posts.append(post)
        ds.append(treated[i])
        
        # 真实ATT = 2.0
        treat_effect = 2.0 if (treated[i] == 1 and post == 1) else 0
        y = unit_fe[i] + 0.5 * t + treat_effect + np.random.randn() * np.sqrt(1 + 0.1*t)
        ys.append(y)

panel_data = pd.DataFrame({
    'id': ids,
    'time': times,
    'y': ys,
    'd': ds,
    'post': posts
})

print(f"\n面板数据: {len(panel_data)} 行, {n_units} 单位, {n_periods} 时期")
print(f"处理组: {treated.sum()} 单位, 控制组: {n_units - treated.sum()} 单位")

# 测试各种 vce 类型
print("\n测试 lwdid() 各 vce 类型:")
print("-" * 60)
results_lwdid = {}
for vce in vce_types:
    try:
        result = lwdid(panel_data, y='y', d='d', ivar='id', tvar='time', post='post', vce=vce)
        results_lwdid[vce] = result
        # 获取ATT和SE
        att = result.effects[0].att if hasattr(result, 'effects') and len(result.effects) > 0 else result.att
        se = result.effects[0].se if hasattr(result, 'effects') and len(result.effects) > 0 else getattr(result, 'se', None)
        if se is None:
            se = result.effects[0].se if hasattr(result, 'effects') else 'N/A'
        print(f"vce='{vce}': ✅ ATT={att:.4f}, SE={se:.4f}" if isinstance(se, float) else f"vce='{vce}': ✅ ATT={att:.4f}")
    except Exception as e:
        print(f"vce='{vce}': ❌ {type(e).__name__}: {e}")

# ============================================================
# 验证结果
# ============================================================
print("\n" + "=" * 60)
print("3. 验证 HC 标准误排序")
print("=" * 60)

if all(v in results_staggered for v in [None, 'hc0', 'hc1', 'hc2', 'hc3', 'hc4']):
    se_hc0 = results_staggered['hc0']['se']
    se_hc1 = results_staggered['hc1']['se']
    se_hc2 = results_staggered['hc2']['se']
    se_hc3 = results_staggered['hc3']['se']
    se_hc4 = results_staggered['hc4']['se']
    
    print(f"SE(HC0) = {se_hc0:.6f}")
    print(f"SE(HC1) = {se_hc1:.6f}")
    print(f"SE(HC2) = {se_hc2:.6f}")
    print(f"SE(HC3) = {se_hc3:.6f}")
    print(f"SE(HC4) = {se_hc4:.6f}")
    
    # 验证排序
    assert se_hc0 <= se_hc1 * 1.001, f"HC0 ({se_hc0}) should be <= HC1 ({se_hc1})"
    print("✅ HC0 <= HC1")
    
    assert se_hc2 <= se_hc3 * 1.001, f"HC2 ({se_hc2}) should be <= HC3 ({se_hc3})"
    print("✅ HC2 <= HC3")

# 验证 robust 和 hc1 相同
print("\n" + "=" * 60)
print("4. 验证 'robust' 和 'hc1' 产生相同结果")
print("=" * 60)

if 'robust' in results_staggered and 'hc1' in results_staggered:
    se_robust = results_staggered['robust']['se']
    se_hc1 = results_staggered['hc1']['se']
    assert np.isclose(se_robust, se_hc1), f"robust ({se_robust}) != hc1 ({se_hc1})"
    print(f"✅ SE(robust) = SE(hc1) = {se_hc1:.6f}")

print("\n" + "=" * 60)
print("所有测试完成!")
print("=" * 60)
