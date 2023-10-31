import numpy as np

class KNN_Classifier():
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None


    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)


    def predict(self, x):
        d = np.sqrt(np.linalg.norm(self.X - x, axis=1))
        index = np.argsort(d)
        neighbor = self.y[index][:self.k]
        return 1 if np.sum(neighbor) * 2 > self.k else 0


model = {
    3: KNN_Classifier(3),
    5: KNN_Classifier(5),
    7: KNN_Classifier(7),
}

n1 = int(input())
train_X = []
train_y = []
for i in range(n1):
    line = input().split()
    train_X.append(list(map(float, line[1:])))
    train_y.append(int(line[0]))

for clf in model.values():
    clf.fit(train_X, train_y)
 
n2 = int(input())
for i in range(n2):
    line = input().split()
    k = int(line[0])
    x = np.array(list(map(float, line[1:])))
    print(model[k].predict(x))

"""
7
1 0 0 0
1 0 0 1
1 0 1 0
1 1 0 0
0 2 3 3
0 4 5 6
0 7 8 9
1
3 0 2 0
"""