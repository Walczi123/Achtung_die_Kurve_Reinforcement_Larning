import sys
sys.path.append('.')
from v2.rl.a2c.a2c import get_a2c_mlp_model
from achtung_tests.test import read_and_show_graph, test_and_save

if __name__ == '__main__':
    # test_and_save((get_a2c_mlp_model(), 'a2c_mlp_v2'))
    read_and_show_graph('a2c_mlp_v2')
