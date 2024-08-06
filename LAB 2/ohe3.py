import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

file_path = '/Users/triahavijayekkumaran/Downloads/Lab Session Data.xlsx'
excel_data = pd.ExcelFile(file_path)
thyrdata = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')

def identify_types(data):
    return data.dtypes

def enc(data):
    enc_data = data.copy()
    labelenc = {}
    onehot_encoders = {}
    
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].dtype == 'mixed':
            data[column] = data[column].astype(str)
            uniqueval = data[column].nunique()
            if uniqueval <= 10:
                le = LabelEncoder()
                enc_data[column] = le.fit_transform(data[column])
                labelenc[column] = le
            else:
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                enc_cols = ohe.fit_transform(data[[column]])
                enc_df = pd.DataFrame(enc_cols, columns=[f"{column}_{cat}" for cat in ohe.categories_[0][1:]])
                enc_data = pd.concat([enc_data.drop(column, axis=1), enc_df], axis=1)
                onehot_encoders[column] = ohe
                
    return enc_data, labelenc, onehot_encoders

def datarange(data):
    return data.describe().T[['min', 'max']]

def miss_vals(data):
    return data.isnull().sum()

def study_outliers(data):
    numeric_data = data.select_dtypes(include=[np.number])
    outliers = {}
    
    for column in numeric_data.columns:
        q1 = numeric_data[column].quantile(0.25)
        q3 = numeric_data[column].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        outliers[column] = numeric_data[(numeric_data[column] < lb) | (numeric_data[column] > ub)]
        
    return outliers

def stats(data):
    numeric_data = data.select_dtypes(include=[np.number])
    means = numeric_data.mean()
    variances = numeric_data.var()
    return means, variances

def plot_outliers(data):
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(numeric_data.columns, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=numeric_data[column])
        plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    plt.show()

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

def normalizedata(data):
    scaler = MinMaxScaler()
    numeric_data = data.select_dtypes(include=[np.number])
    scaled_data = scaler.fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
    return pd.concat([data.drop(numeric_data.columns, axis=1), scaled_df], axis=1)

data_types = identify_types(thyrdata)
enc_data, labelenc, onehot_encoders = enc(thyrdata)
data_range = datarange(thyrdata)
missing_values = miss_vals(thyrdata)
outliers_dict = study_outliers(thyrdata)
means, variances = stats(thyrdata)

print("Data Types:\n", data_types)
print("\nData Range:\n", data_range)
print("\nMissing Values:\n", missing_values)
print("\nOutliers:\n", {key: len(value) for key, value in outliers_dict.items()})
print("\nMeans:\n", means)
print("\nVariances:\n", variances)

plot_outliers(thyrdata)

imp_data = impute_values(thyrdata)

missing_vals_after = miss_vals(imp_data)

norm_data = normalizedata(imp_data)

print("\nNormalized Data:\n", norm_data.head())
