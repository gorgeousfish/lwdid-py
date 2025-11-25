import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from lwdid import lwdid


def test_visualization_elements_and_gid_checks():
    here = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(here, 'data', 'smoking.csv'))

    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='detrend', vce='robust'
    )

    # plotting with treated average (no gid)
    fig = res.plot(gid=None, graph_options={'dpi': 100})
    assert fig is not None

    # invalid gid
    try:
        res.plot(gid='NonExist')
        assert False, 'expected InvalidParameterError for missing gid'
    except Exception as e:
        assert "gid 'NonExist' not found" in str(e)


