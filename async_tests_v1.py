import multiprocessing
import time
import tqdm

from v1.rl.a2c.a2c import get_a2c_model
from v1.rl.dqn.dqn import get_dqn_cnn_model, get_dqn_mlp_model

from tests.test import test_and_save

def generate_instances():
    result = []

    result.append((get_dqn_cnn_model(), 'dqn_cnn'))
    result.append((get_dqn_mlp_model(), 'dqn_mpl'))  
    result.append((get_a2c_model(), 'a2c'))
    # result.append((get_cnn_model(), 'cnn'))

    return result


def run_test(test):
    print(f'start of {test[1]}')
    test_and_save(test)

def run_tests():
    iterable = generate_instances()
    print(iterable)

    start_time = time.time()

    # test_and_save(iterable[0])

    max_cpu = multiprocessing.cpu_count()
    p = multiprocessing.Pool(int(max_cpu)-2)
    # for _ in tqdm.tqdm(p.imap_unordered(run_test, iterable), total=len(iterable)):
    #     pass
    p.map_async(run_test, iterable)

    p.close()
    p.join()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run_tests()
