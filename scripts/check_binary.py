import numpy as np

N=1000
D=8
X = np.fromfile("data/test.bin", dtype=np.float64)
X = X.reshape(N, D)

print(X[:5])
print(X.shape)