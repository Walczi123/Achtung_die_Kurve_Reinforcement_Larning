import sys
sys.path.append('.')
from v1.rl.a2c.a2c import get_a2c_model
from tests.test import test_and_save

if __name__ == '__main__':
    test_and_save((get_a2c_model(), 'a2c'))
