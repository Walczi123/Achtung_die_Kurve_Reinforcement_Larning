import sys
sys.path.append('.')
from v2.rl.dqn.dqn import get_dqn_mlp_model
from achtung_tests.test import read_and_show_graph, test_and_save

if __name__ == '__main__':
    test_and_save((get_dqn_mlp_model(), 'dqn_mlp_v2'))
    read_and_show_graph('dqn_mlp_v2')