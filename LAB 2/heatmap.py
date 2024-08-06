import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

def calculate_jc_smc(data):
    jc_matrix = np.zeros((len(data), len(data)))
    smc_matrix = np.zeros((len(data), len(data)))

    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                vector1 = data.iloc[i]
                vector2 = data.iloc[j]
                
                f11 = np.sum((vector1 == 1) & (vector2 == 1))
                f01 = np.sum((vector1 == 0) & (vector2 == 1))
                f10 = np.sum((vector1 == 1) & (vector2 == 0))
                f00 = np.sum((vector1 == 0) & (vector2 == 0))
                
                jc = f11 / (f01 + f10 + f11)
                smc = (f11 + f00) / (f00 + f01 + f10 + f11)
                
                jc_matrix[i, j] = jc
                smc_matrix[i, j] = smc

    return jc_matrix, smc_matrix

def cosinesimilarity(data):
    return cosine_similarity(data)

thyrdata = impute_values(thyrdata)
binary_data, labelenc, onehot_encoders = enc(thyrdata)
binary_data = binary_data.head(20)

jc_matrix, smc_matrix = calculate_jc_smc(binary_data)

cosine_matrix = cosinesimilarity(binary_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(jc_matrix, ax=axes[0], cmap='coolwarm', annot=True, fmt=".2f")
axes[0].set_title('Jaccard Coefficient')

sns.heatmap(smc_matrix, ax=axes[1], cmap='coolwarm', annot=True, fmt=".2f")
axes[1].set_title('Simple Matching Coefficient')

sns.heatmap(cosine_matrix, ax=axes[2], cmap='coolwarm', annot=True, fmt=".2f")
axes[2].set_title('Cosine Similarity')

plt.tight_layout()
plt.show()
