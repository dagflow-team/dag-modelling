from dag_modelling.bundles.load_parameters import load_parameters
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph

cfg1 = {
        'parameters': {
            'var1': 1.0,
            'var2': 1.0,
            'sub1': {
                'var3': 2.0
                }
            },
        'format': 'value',
        'state': 'variable',
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-1'
                },
            'var2': 'simple label 2',
            },
        }
cfg1a = {
        'parameters': {
            'var1': 1.0,
            'var2': 1.0,
            'sub1': {
                'var3': 2.0
                }
            },
        'format': 'value',
        'state': 'fixed',
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-1'
                },
            'var2': 'simple label 2',
            },
        }

cfg2 = {
        'parameters': {
            'var1': (1.0, 1.0, 0.1),
            'var2': (1.0, 2.0, 0.1),
            'sub1': {
                'var3': (2.0, 1.0, 0.1)
                }
            },
        'path': 'sub.folder',
        'format': ('value', 'central', 'sigma_absolute'),
        'state': 'fixed',
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-2'
                },
            'var2': 'simple label 2'
            },
        }

cfg3 = {
        'parameters': {
            'var1': [1.0, 1.0, 0.1],
            'var2': (1.0, 2.0, 0.1),
            'sub1': {
                'var3': (2.0, 3.0, 0.1)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-3'
                },
            'var2': 'simple label 2'
            },
        'format': ['value', 'central', 'sigma_relative'],
        'state': 'fixed',
        }

cfg4 = {
        'parameters': {
            'var1': (1.0, 1.0, 10),
            'var2': (1.0, 2.0, 10),
            'sub1': {
                'var3': (2.0, 3.0, 10)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        'format': ('value', 'central', 'sigma_percent'),
        'state': 'variable',
        }

cfg5 = {
        'parameters': {
            'var1': (1.0, 10),
            'var2': (2.0, 10),
            'sub1': {
                'var3': (3.0, 10)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        'format': ('central', 'sigma_percent'),
        'state': 'variable',
        }

from pprint import pprint
def test_load_parameters_v01():
    cfgs = (cfg1, cfg1a, cfg2, cfg3, cfg4, cfg5)
    with Graph(close_on_exit=True) as g:
        for i, cfg in enumerate(cfgs):
            vars = load_parameters(cfg)
            # print(cfg['state'])
            print(i, end=' ')
            pprint(vars.object)

    savegraph(g, 'output/test_load_parameters.pdf', show='all')
