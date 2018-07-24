import numpy as np
import h5py
import matplotlib.pyplot as plt

def zero_pad(X, pad):

	X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

	return X_pad

def conv_single_step(a_slice_prev, W, b):
	s = a_slice_prev * W
	Z = np.sum(s)
	Z = Z + np.float(b)

	return Z

def conv_forward(A_prev, W, b, hparameters):

	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

	(f, f, n_C_prev, n_C) = W.shape

	stride = hparameters["stride"]
	pad = hparameters["pad"]

	n_H = int(((n_H_prev - f + (2 * pad)) / stride)) + 1
	n_W = int(((n_W_prev - f + (2 * pad)) / stride)) + 1

	Z = np.zeros((m, n_H, n_W, n_C))
	
	A_prev_pad = zero_pad(A_prev, pad)

	for i in range(m):
		a_prev_pad = A_prev_pad[i]
		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):
					
					vert_start = h * stride
					vert_end = vert_start + f
					horiz_start = w * stride
					horiz_end = horiz_start + f

					a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
					Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

	assert(Z.shape == (m, n_H, n_W, n_C))

	cache = (A_prev, W, b, hparameters)
	return Z, cache

def conv_backward(dZ, cache):

	(A_prev, W, b, hparameters) = cache

	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

	(f, f, n_C_prev, n_C) = W.shape

	stride = hparameters["stride"]
	pad = hparameters["pad"]

	(m, n_H, n_W, n_C) = dZ.shape

	dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
	dW = np.zeros((f, f, n_C_prev, n_C))
	db = np.zeros((1, 1, 1, n_C))

	A_prev_pad = zero_pad(A_prev, pad)
	dA_prev_pad = zero_pad(dA_prev, pad)

	for i in range(m):

 		a_prev_pad = A_prev_pad[i]
 		da_prev_pad = dA_prev_pad[i]

 		for h in range(n_H):
 			for w in range(n_W):
 				for c in range(n_C):

 					vert_start = h
 					vert_end = vert_start + f
 					horiz_start = w
 					horiz_end = horiz_start + f

 					a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

 					da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h ,w, c]
 					dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
 					db[:, :, :, c] += dZ[i, h, w, c]

 		dA_prev[i, :, :, :] = da_prev_pad[pad: -pad, pad: -pad, :]

	assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev)) 

	return dA_prev, dW, db

def pool_forward(A_prev, hparameters, mode = "max"):

	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

	f = hparameters["f"]
	stride = hparameters["stride"]

	n_H = int(1 + (n_H_prev - f) / stride)
	n_W = int(1 + (n_W_prev - f) / stride)
	n_C = n_C_prev

	A = np.zeros((m, n_H, n_W, n_C))

	for i in range(m):
		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):

					vert_start = h * stride
					vert_end = vert_start + f
					horiz_start = w * stride
					horiz_end = horiz_start + f

					a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start: horiz_end, :]

					if mode == "max":
						A[i, h, w, c] = np.max(a_prev_slice)
					elif mode == "average":
						A[i, h, w, c] = np.mean(a_prev_slice)

	cache = (A_prev, hparameters)

	assert(A.shape == (m, n_H, n_W, n_C))

	return A, cache

def create_mask_from_window(x):

	mask = np.random.randn(x.shape[0], x.shape[1])
	X = np.max(x)

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if x[i, j] == X:
				mask[i, j] = True
			elif x[i, j] != X:
				mask[i, j] = False

	return mask

def distribute_value(dz, shape):

	(n_H, n_W) = shape
	average = dz / (n_H * n_W)
	a = np.array([[average for j in range(n_W)] for i in range(n_H)])
	return a


def pool_backward(dA, cache, mode = "max"):

	(A_prev, hparameters) = cache

	stride = hparameters["stride"]
	f = hparameters["f"]

	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	(m, n_H, n_W, n_C) = dA.shape

	dA_prev = np.zeros(A_prev.shape)

	for i in range(m):

		a_prev = A_prev[i]

		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):

					vert_start = h
					vert_end = vert_start + f
					horiz_start = w
					horiz_end = horiz_start + f

					if mode == "max":

						a_prev_slice = a_prev[vert_start: vert_end, horiz_start: horiz_end, c]

						mask = create_mask_from_window(a_prev_slice)
						dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += (mask * dA[i, h, w, c])
					elif mode == "average":

						a = dA[i, h, w, c]
						shape = (f, f)
						dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(a, shape)

	print("dA_prev.shape =", A_prev.shape)
	assert(dA_prev.shape == A_prev.shape)  

	return dA_prev

def test():

	###START CODE###

	np.random.seed(1)
	x = np.random.randn(4, 3, 3, 2)
	x_pad = zero_pad(x, 2)
	print('x.shape =', x.shape)
	print("x_pad.shape =", x_pad.shape)
	print("x[1, 1] =", x[1, 1])
	print("x_pad[1, 1] =", x_pad[1, 1])

	###END CODE###

	###START CODE conv_single_step###

	a_slice_prev = np.random.randn(4, 4, 3)
	W = np.random.randn(4, 4, 3)
	b = np.random.randn(1, 1, 1)

	Z = conv_single_step(a_slice_prev, W, b)
	print("Z =", Z)

	###END CODE###

	###START CODE conv_forward###

	A_prev = np.random.randn(10, 4, 4, 3)
	W = np.random.randn(2, 2, 3, 8)
	b = np.random.randn(1, 1, 1, 8)
	hparameters = {"pad": 2, "stride": 2}

	Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
	print("Z's mean = ", np.mean(Z))
	print("Z[3 , 2, 1] =", Z[3, 2, 1])
	print("cache_conv [0][1][2][3] =", cache_conv[0][1][2][3])
	###END CODE###

	##START POOLING LAYER###

	A_prev = np.random.randn(2, 4, 4, 3)
	hparameters = {"stride": 2, "f": 3}

	A, cache = pool_forward(A_prev, hparameters)
	print("mode = max")
	print("A = ", A)
	print()

	A, cache = pool_forward(A_prev, hparameters, mode = "average")
	print("mode = average")
	print("A =", A)

	######

	###START BACKPROPAGATION CONV###

	dA, dW, db = conv_backward(Z, cache_conv)
	print("dA_mean =", np.mean(dA))
	print("dW_mean =", np.mean(dW))
	print("db_mean =", np.mean(db))

	##END CODE###

	###START create_mask###
	x = np.random.randn(2, 3)
	mask = create_mask_from_window(x)
	print("x = ", x)
	print("mask = ", mask)
	###END CODE###

	###START distributed_value###

	a = distribute_value(2, (2, 2))
	print("distributed value =", a)

	###END CODE###

	A_prev = np.random.randn(5, 5, 3, 2)
	hparameters = {"stride" : 1, "f": 2}
	A, cache = pool_forward(A_prev, hparameters)
	dA = np.random.randn(5, 4, 2, 2)

	dA_prev = pool_backward(dA, cache, mode = "max")
	print("mode = max")
	print('mean of dA = ', np.mean(dA))
	print('dA_prev[1,1] = ', dA_prev[1,1])  
	print()

if __name__ == '__main__':
	test()
