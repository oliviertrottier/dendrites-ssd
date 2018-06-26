from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import concurrent.futures

# what are your inputs, and what operation do you want to
# perform on each input. For example...

def processInput(i,j):
    a=1
    print(i, j, a)
    return (i * j, np.sum(np.random.rand(1, int(j * 1000)) < 0.5))
#
# num_cores = multiprocessing.cpu_count()
# args = zip(range(10), np.random.randint(0, high=20, size=(10,)).tolist())
# result = Parallel(n_jobs=num_cores)(delayed(processInput)(arg) for arg in args)
# print(result)
# num_cores = multiprocessing.cpu_count()
# args = zip(range(10), np.random.randint(0,high=20,size=(10,)).tolist())
# results = Parallel(n_jobs=num_cores)(delayed(main)(arg) for arg in args)
# results = main()
# print(results)


# def processInput2(args):
#     i = args[0]
#     j = args[1]
#     print(i, j, a)
#     return (i * j, np.sum(np.random.rand(1, int(j * 1000)) < 0.5))
#
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     args = zip(range(10), np.random.randint(0, high=20, size=(10,)).tolist())
#     for i,j in executor.submit(processInput2, args):
#         print(results.result())
# pass

# import multiprocessing
# from itertools import product
#
#
# def merge_names(a, b):
#     return '{} & {}'.format(a, b)
#
# def main():
#     N_tasks=1000
#     args = zip(range(N_tasks), np.random.randint(0, high=20, size=(N_tasks,)).tolist())
#     names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
#     with multiprocessing.Pool(processes=3) as pool:
#         #results = pool.starmap(merge_names, product(names, repeat=2))
#         results2 = pool.starmap(processInput, args)
#     print(results2)
#
# main()
#
# from multiprocessing import Pool
#
#
# def doubler(number):
#     return number * 2
#
# if __name__ == '__main__':
#     numbers = [5, 10, 20]
#     pool = Pool(processes=3)
#     results=pool.map(doubler, numbers)
#     print(results)

class tree(object):

    def __init__(self, a):
        self.a=a

s=tree(10)
print(s.a)
s.b=2
print(s.b)