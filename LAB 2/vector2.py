import pandas as pd
import numpy as np

file_path = '/Users/triahavijayekkumaran/Downloads/Lab Session Data.xlsx'
exceldata = pd.ExcelFile(file_path)

purchase_data = pd.read_excel(file_path, sheet_name='Purchase data')
A = purchase_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = purchase_data[['Payment (Rs)']].values

a_pinv = np.linalg.pinv(A)
X = a_pinv @ C

print("Model vector X (Cost of each product available for sale):", X.flatten())