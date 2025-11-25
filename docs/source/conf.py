# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# RST替换定义已移除 - 直接在文档中使用完整文本
# rst_prolog = ""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lwdid'
copyright = '2025, Xuanyu Cai, Wenli Xu'
author = 'Xuanyu Cai, Wenli Xu'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # 自动从docstrings生成API文档
    'sphinx.ext.autosummary',       # 自动生成API摘要
    'sphinx.ext.napoleon',          # 支持NumPy和Google风格的docstrings
    'sphinx.ext.viewcode',          # 添加源代码链接
    'sphinx.ext.intersphinx',       # 链接到其他项目的文档
    'sphinx.ext.mathjax',           # 数学公式支持
    'sphinx_autodoc_typehints',     # 类型提示支持
    'sphinx_copybutton',            # 代码块复制按钮
    'myst_parser',                  # Markdown支持
]

# 自动生成API摘要
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# 避免property重复文档化
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# 防止单字母参数名被误识别为交叉引用
suppress_warnings = ['ref.python']

# Napoleon设置 - 支持NumPy风格的docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping - 链接到其他Python包文档
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
}

# 模板和静态文件路径
templates_path = ['_templates']
exclude_patterns = []

# 支持的文件格式
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# ReadTheDocs主题配置
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# 侧边栏配置
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# 代码高亮
pygments_style = 'sphinx'

# 文档标题
html_title = f'{project} {version}'
html_short_title = project

# 显示源文件链接
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Markdown相关设置
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]