import numpy as np

# Tạo mảng 2D với 2 dòng, 3 cột
X = np.array([[1.0, 2.0, 3.0], 
              [1.0, 2.0, 3.0]])
print(X)
print("Shape ban đầu:", X.shape)

# Reshape thành (6, 1)
X = X.reshape(-1, 1)
print("\nSau reshape (-1, 1):")
print(X)
print("Shape mới:", X.shape)
