import sys
sys.path.append('.')
from v2.rl.dqn.dqn import get_dqn_mlp_model
from tests.test import test_and_save

if __name__ == '__main__':
    test_and_save((get_dqn_mlp_model(), 'dqn_mlp'))
