# -*- coding: utf-8 -*-
"""
报告和可视化模块测试

Task 7.4 & 11.2: 测试性能报告和可视化功能
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

# 添加路径
reports_path = Path(__file__).parent.parent / 'reports'
sys.path.insert(0, str(reports_path))


class TestPerformanceReport:
    """性能报告测试"""
    
    def test_generate_comparison_table(self):
        """测试生成对比表格"""
        from performance_report import generate_comparison_table, PAPER_TABLE2_BENCHMARKS
        
        # 模拟结果
        results = [
            {'scenario': '1', 'estimator': 'OLS_Demean', 'n_units': 20,
             'bias': 2.50, 'sd': 5.20, 'rmse': 5.80, 'coverage': 0.94},
            {'scenario': '1', 'estimator': 'OLS_Detrend', 'n_units': 20,
             'bias': 0.20, 'sd': 5.60, 'rmse': 5.60, 'coverage': 0.95},
        ]
        
        df = generate_comparison_table(results)
        
        assert len(df) == 2
        assert 'Bias' in df.columns
        assert 'Paper_Bias' in df.columns
        assert 'Bias_Diff' in df.columns
    
    def test_format_report(self):
        """测试格式化报告"""
        from performance_report import generate_comparison_table, format_report
        
        results = [
            {'scenario': '1', 'estimator': 'OLS_Demean', 'n_units': 20,
             'bias': 2.50, 'sd': 5.20, 'rmse': 5.80, 'coverage': 0.94},
        ]
        
        df = generate_comparison_table(results)
        report = format_report(df)
        
        assert "Performance Comparison Report" in report
        assert "Bias" in report
        assert "PASS" in report or "FAIL" in report
    
    def test_save_report(self):
        """测试保存报告"""
        from performance_report import generate_comparison_table, save_report
        
        results = [
            {'scenario': '1', 'estimator': 'OLS_Demean', 'n_units': 20,
             'bias': 2.50, 'sd': 5.20, 'rmse': 5.80, 'coverage': 0.94},
        ]
        
        df = generate_comparison_table(results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_report(df, Path(tmpdir), "test_report")
            
            assert paths['csv'].exists()
            assert paths['txt'].exists()
            
            # 验证 CSV 内容
            loaded_df = pd.read_csv(paths['csv'])
            assert len(loaded_df) == 1


class TestVisualization:
    """可视化模块测试"""
    
    @pytest.fixture
    def sample_results(self):
        """示例结果数据"""
        return [
            {'scenario': '1', 'estimator': 'OLS_Demean', 'n_units': 20,
             'bias': 2.50, 'sd': 5.20, 'rmse': 5.80, 'coverage': 0.94},
            {'scenario': '1', 'estimator': 'OLS_Detrend', 'n_units': 20,
             'bias': 0.20, 'sd': 5.60, 'rmse': 5.60, 'coverage': 0.95},
            {'scenario': '2', 'estimator': 'OLS_Demean', 'n_units': 20,
             'bias': 2.40, 'sd': 5.30, 'rmse': 5.85, 'coverage': 0.93},
            {'scenario': '2', 'estimator': 'OLS_Detrend', 'n_units': 20,
             'bias': 0.15, 'sd': 5.70, 'rmse': 5.70, 'coverage': 0.96},
        ]
    
    @pytest.fixture
    def sample_att_estimates(self):
        """示例 ATT 估计值"""
        np.random.seed(42)
        return np.random.normal(4.0, 1.5, 100)
    
    def test_has_matplotlib_flag(self):
        """测试 matplotlib 可用性标志"""
        from visualization import HAS_MATPLOTLIB
        # 只检查标志存在
        assert isinstance(HAS_MATPLOTLIB, bool)
    
    @pytest.mark.skipif(
        not __import__('visualization', fromlist=['HAS_MATPLOTLIB']).HAS_MATPLOTLIB,
        reason="matplotlib 未安装"
    )
    def test_plot_att_distribution(self, sample_att_estimates):
        """测试 ATT 分布图"""
        from visualization import plot_att_distribution
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        result_ax = plot_att_distribution(sample_att_estimates, true_att=4.0, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    @pytest.mark.skipif(
        not __import__('visualization', fromlist=['HAS_MATPLOTLIB']).HAS_MATPLOTLIB,
        reason="matplotlib 未安装"
    )
    def test_plot_bias_by_sample_size(self):
        """测试 Bias 随样本量变化图"""
        from visualization import plot_bias_by_sample_size
        import matplotlib.pyplot as plt
        
        results = [
            {'n_units': 20, 'bias': 2.5, 'estimator': 'OLS'},
            {'n_units': 50, 'bias': 1.5, 'estimator': 'OLS'},
            {'n_units': 100, 'bias': 0.8, 'estimator': 'OLS'},
        ]
        
        fig, ax = plt.subplots()
        result_ax = plot_bias_by_sample_size(results, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    @pytest.mark.skipif(
        not __import__('visualization', fromlist=['HAS_MATPLOTLIB']).HAS_MATPLOTLIB,
        reason="matplotlib 未安装"
    )
    def test_plot_coverage_by_scenario(self, sample_results):
        """测试 Coverage 随场景变化图"""
        from visualization import plot_coverage_by_scenario
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        result_ax = plot_coverage_by_scenario(sample_results, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    @pytest.mark.skipif(
        not __import__('visualization', fromlist=['HAS_MATPLOTLIB']).HAS_MATPLOTLIB,
        reason="matplotlib 未安装"
    )
    def test_plot_rmse_comparison(self, sample_results):
        """测试 RMSE 对比图"""
        from visualization import plot_rmse_comparison
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        result_ax = plot_rmse_comparison(sample_results, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    @pytest.mark.skipif(
        not __import__('visualization', fromlist=['HAS_MATPLOTLIB']).HAS_MATPLOTLIB,
        reason="matplotlib 未安装"
    )
    def test_create_summary_figure(self, sample_results, sample_att_estimates):
        """测试创建汇总图表"""
        from visualization import create_summary_figure
        import matplotlib.pyplot as plt
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.png"
            
            fig = create_summary_figure(
                sample_results,
                att_estimates=sample_att_estimates,
                true_att=4.0,
                output_path=output_path,
            )
            
            assert fig is not None
            assert output_path.exists()
            plt.close(fig)


class TestIntegration:
    """集成测试：报告 + 可视化"""
    
    @pytest.mark.skipif(
        not __import__('visualization', fromlist=['HAS_MATPLOTLIB']).HAS_MATPLOTLIB,
        reason="matplotlib 未安装"
    )
    def test_full_report_workflow(self):
        """测试完整报告工作流"""
        from performance_report import generate_comparison_table, save_report
        from visualization import create_summary_figure
        import matplotlib.pyplot as plt
        
        # 模拟 Monte Carlo 结果
        results = [
            {'scenario': '1', 'estimator': 'OLS_Demean', 'n_units': 20,
             'bias': 2.44, 'sd': 5.30, 'rmse': 5.83, 'coverage': 0.93},
            {'scenario': '1', 'estimator': 'OLS_Detrend', 'n_units': 20,
             'bias': 0.15, 'sd': 5.67, 'rmse': 5.67, 'coverage': 0.96},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 1. 生成对比表格
            df = generate_comparison_table(results)
            assert len(df) == 2
            
            # 2. 保存报告
            paths = save_report(df, tmpdir, "mc_report")
            assert paths['csv'].exists()
            assert paths['txt'].exists()
            
            # 3. 生成可视化
            fig = create_summary_figure(
                results,
                output_path=tmpdir / "summary.png",
            )
            assert (tmpdir / "summary.png").exists()
            plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
