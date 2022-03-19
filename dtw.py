import nibabel as nb
import pandas as pd
import numpy as np
from numpy import zeros
from numba import jit, prange
import time


INF = 2e32


@jit(nopython=True, parallel=True)
def init_d0(D0, r, c, w):
	for i in prange(1, r + 1):
		D0[i, max(1, i - w): min(c + 1, i + w + 1)] = 0
	D0[0, 0] = 0
	return D0


@jit(nopython=True, parallel=True)
def init_d1(D1, r, c, w, x, y):
	for i in prange(r):
		for j in prange(c):
			if (w == INF or (max(0, i - w) <= j <= min(c, i + w))):
				D1[i, j] = np.abs(x[i] - y[j])
	return D1


@jit(nopython=True)
def calc_d1(D0, D1, r, c, w, s, warp):
	jrange = prange(c)
	for i in prange(r):
		if not w == INF:
			jrange = prange(max(0, i - w), min(c, i + w + 1))
		for j in jrange:
			min_list = [D0[i, j]]
			for k in prange(1, warp + 1):
				i_k = min(i + k, r)
				j_k = min(j + k, c)
				min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
			D1[i, j] += min(min_list)
	return D1


def accelerated_dtw(x, y, warp=1, w=INF, s=1.0):
	assert len(x)
	assert len(y)
	assert w == INF or (w >= abs(len(x) - len(y)))
	assert s > 0
	r, c = len(x), len(y)
	if not w == INF:
		D0 = full((r + 1, c + 1), INF)
		D0 = init_d0(D0, r, c, w)
	else:
		D0 = zeros((r + 1, c + 1))
		D0[0, 1:] = INF
		D0[1:, 0] = INF
	D1 = D0[1:, 1:]
	D1 = init_d1(D1, r, c, w, x, y)
	D1 = calc_d1(D0, D1, r, c, w, s, warp)
	return D1[-1, -1]


"""
The fastest DTW for bold signals... so far
O(n^3)
TO DO: More parallelization
"""
def dynamic_time_warping(data):
	mean = np.mean(data, axis=0)
	dists = []
	for signal in data:
		dist = accelerated_dtw(mean, signal)
		dists.append(dist)
	return dists


def main():
	names = ['x', 'y', 'z']+list(range(0, 268))
	df = pd.read_csv('test.csv', sep=' ', names=names)

	data_df = df.drop(df.columns[[0, 1, 2]], axis=1)
	data = data_df.to_numpy()

	start_time = time.time()
	dists = dynamic_time_warping(data)
	print('%s seconds' % (time.time() - start_time))
	df.insert(3, 'dtwdist', dists)
	print(df.head())


if __name__ == '__main__':
	main()
