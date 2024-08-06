import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

file_path = '/Users/triahavijayekkumaran/Downloads/Lab Session Data.xlsx'
excel_data = pd.ExcelFile(file_path)
thyrdata = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')
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

def cosinesimilarity(data):
    # Select the first two observation vectors
    vector1 = data.iloc[0].values.reshape(1, -1)
    vector2 = data.iloc[1].values.reshape(1, -1)
    
    # Calculate cosine similarity
    cos_sim = cosine_similarity(vector1, vector2)
    
    return cos_sim[0][0]

thyrdata = impute_values(thyrdata)

enc_data, labelenc, onehot_encoders = enc(thyrdata)

cosinesim = cosinesimilarity(enc_data)

print(f"Cosine Similarity: {cosinesim}")
