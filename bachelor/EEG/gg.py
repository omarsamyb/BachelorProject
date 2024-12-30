import numpy as np

x = [[[2, 2, 2, 2],
      [3, 3, 3, 3],
      [4, 4, 4, 4]],
     [[2, 2, 2, 2],
      [3, 3, 3, 3],
      [4, 4, 4, 4]]]
x = np.array(x)
print(x.shape)
print(x)
print("$$$$$$4")
x = np.mean(x, axis=2, keepdims=True)
print(x.shape)
print(x)
