import pandas as pd
import numpy as np


file_path = '/Users/triahavijayekkumaran/Downloads/Lab Session Data.xlsx'
exceldata = pd.ExcelFile(file_path)


purchase_data = pd.read_excel(file_path, sheet_name='Purchase data')


A = purchase_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = purchase_data[['Payment (Rs)']].values

dim = A.shape[1]

num_v = A.shape[0]

rank = np.linalg.matrix_rank(A)

a_pinv = np.linalg.pinv(A)

costs = a_pinv @ C

# Print the results
print("Dimensionality of the vector space:", dim)
print("Number of vectors in the vector space:", num_v)
print("Rank of Matrix A:", rank)
print("Cost of each product available for sale (Candies, Mangoes, Milk Packets):", costs.flatten())
