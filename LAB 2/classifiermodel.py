import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

fp = '/Users/triahavijayekkumaran/Downloads/Lab Session Data.xlsx'
excel_data = pd.ExcelFile(fp)

p_d = pd.read_excel(fp, sheet_name='Purchase data')

mod_data = p_d[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

mod_data['Category'] = np.where(mod_data['Payment (Rs)'] > 200, 'RICH', 'POOR')

X = mod_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
y = mod_data['Category'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=30)

cfr = LogisticRegression()
cfr.fit(X_train, y_train)

y_pred = cfr.predict(X_test)

print(classification_report(y_test, y_pred))

mod_data.head()
