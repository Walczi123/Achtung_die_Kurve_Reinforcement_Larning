import sys
sys.path.append('.')
from v1.rl.a2c.a2c import get_a2c_cnn_model
from tests.test import read_and_show_graph, test_and_save

if __name__ == '__main__':
    test_and_save((get_a2c_cnn_model(), 'a2c_cnn'))
    read_and_show_graph('a2c_cnn')
