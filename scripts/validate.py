import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

N = 1000
D = 8

X = np.fromfile("data/test.bin", dtype=np.float64)
X = X.reshape(N, D)

Y = np.fromfile("out.bin", dtype=np.float64)
Y = Y.reshape(N, D)

scaler = StandardScaler()
Y_ref = scaler.fit_transform(X)

diff = np.abs(Y - Y_ref)

print("Max error :", diff.max())
print("Mean error:", diff.mean())

Y_minmax = np.fromfile("out.bin", dtype=np.float64).reshape(N, D)
scaler2 = MinMaxScaler()
Y_ref_minmax = scaler2.fit_transform(X)
diff2 = np.abs(Y_minmax - Y_ref_minmax)
print("MinMax Max error :", diff2.max())
print("MinMax Mean error:", diff2.mean())
