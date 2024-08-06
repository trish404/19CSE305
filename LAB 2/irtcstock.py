import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

fp = '/Users/triahavijayekkumaran/Downloads/Lab Session Data.xlsx'
excel_data = pd.ExcelFile(fp)

stock = pd.read_excel(fp, sheet_name='IRCTC Stock Price')

data = stock['Price'].dropna()
meanprice = statistics.mean(data)
print(meanprice)
varianceprice = statistics.variance(data)
print(varianceprice)

wednesdayp = stock[stock['Day'] == 'Wed']['Price']
meanwednesdayp = statistics.mean(wednesdayp)

aprilp = stock[stock['Month'] == "Apr"]['Price']
meanaprilp = statistics.mean(aprilp)

chgdata = stock['Chg%']
loss = len(chgdata[chgdata < 0]) / len(chgdata)

wednesdaychg = stock[stock['Day'] == 'Wed']['Chg%'].dropna()
profitwed = len(wednesdaychg[wednesdaychg > 0]) / len(wednesdaychg)

profitgwednesday = (profitwed * (len(wednesdaychg) / len(chgdata)))

plt.figure(figsize=(10, 6))
plt.scatter(stock['Day'], stock['Chg%'], alpha=0.5)
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Chg% vs Day of the Week')
plt.show()

(meanprice, varianceprice, meanwednesdayp, meanaprilp, loss, profitwed, profitgwednesday)
