import numpy as np

N, M = map(int, input().split())
train_data = np.array([list(map(float, input().split())) for _ in range(N)])
x_train = np.column_stack([np.ones(N), train_data[:, 0], train_data[:, 0]**2, train_data[:, 0]**3])
y_train = train_data[:, 1]
w = np.linalg.pinv(x_train).dot(y_train)
test_data = np.array([float(input()) for _ in range(M)])
X_test = np.column_stack([np.ones(M), test_data, test_data**2, test_data**3])
y_pred = X_test.dot(w)

for y in y_pred:
    print(y)