import pandas as pd
import numpy as np

file_path = '/Users/triahavijayekkumaran/Downloads/Lab Session Data.xlsx'
excel_data = pd.ExcelFile(file_path)
thyrdata = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')

def impute_values(data):
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype == 'object':
                mode = data[column].mode()[0]
                data[column].fillna(mode, inplace=True)
            else:
                outliers = study_outliers(data)
                if len(outliers[column]) > 0:
                    median = data[column].median()
                    data[column].fillna(median, inplace=True)
                else:
                    mean = data[column].mean()
                    data[column].fillna(mean, inplace=True)
    return data

def calculate_similarity_measures(data):
    vector1 = data.iloc[0]
    vector2 = data.iloc[1]
    
    binary_attr = [col for col in data.columns if set(data[col].unique()) <= {0, 1}]
    binary_v1 = vector1[binary_attr]
    binary_v2 = vector2[binary_attr]
    
    f11 = np.sum((binary_v1 == 1) & (binary_v2 == 1))
    f01 = np.sum((binary_v1 == 0) & (binary_v2 == 1))
    f10 = np.sum((binary_v1 == 1) & (binary_v2 == 0))
    f00 = np.sum((binary_v1 == 0) & (binary_v2 == 0))
    
    jc = f11 / (f01 + f10 + f11)
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    
    return jc, smc

thyrdata = impute_values(thyrdata)

binary_data = pd.get_dummies(thyrdata, drop_first=True)

jc, smc = calculate_similarity_measures(binary_data)

print(f"Jaccard Coefficient (JC): {jc}")
print(f"Simple Matching Coefficient (SMC): {smc}")
