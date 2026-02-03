"""验证HC标准误实现状态"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')

# 1. 检查函数是否存在
print("=" * 50)
print("1. 检查核心函数是否存在")
print("=" * 50)
try:
    from lwdid.staggered.estimation import _compute_leverage, _compute_hc_variance, _compute_hc1_variance
    print("✅ _compute_leverage 存在")
    print("✅ _compute_hc_variance 存在")
    print("✅ _compute_hc1_variance 存在")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# 2. 测试所有HC类型
print("\n" + "=" * 50)
print("2. 测试所有HC类型")
print("=" * 50)
from lwdid import lwdid

np.random.seed(42)
n = 100
data = pd.DataFrame({
    'id': np.repeat(range(20), 5),
    'time': np.tile(range(5), 20),
    'y': np.random.randn(n),
    'd': np.repeat([0]*10 + [1]*10, 5),
    'g': np.repeat([0]*10 + [3]*10, 5)
})

for vce in [None, 'hc0', 'hc1', 'robust', 'hc2', 'hc3', 'hc4']:
    try:
        result = lwdid(data, 'y', 'd', 'g', 'id', 'time', vce=vce)
        print(f"✅ vce={str(vce):8s}: ATT={result.att:.4f}, SE={result.se:.4f}")
    except Exception as e:
        print(f"❌ vce={str(vce):8s}: {e}")

print("\n验证完成!")
