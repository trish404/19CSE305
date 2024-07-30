import numpy as np
def mat_pw(A, m):
    if m < 1:
        return "The exponent must be a positive integer."
    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        return "The matrix must be square."
    res = np.linalg.matrix_power(A, m)
    return res

n = int(input("Enter the size of the matrix (n x n): "))

print(f"Enter the {n}x{n} matrix:")
A = []
for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

m = int(input("Enter the power to which the matrix should be raised: "))

res = mat_pw(A, m)
print("The resulting is:")
print(res)
