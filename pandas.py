import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Data
data = {
    'Month': ['January', 'February', 'March', 'April'],
    'TargetSales': [50000, 55000, 60000, 65000],
    'ActualSales': [48000, 60000, 58000, 70000]
}
df = pd.DataFrame(data)
df['Performance'] = (df['ActualSales'] / df['TargetSales']) * 100

# Summary
print(df)
print("Overall Performance:", round((df['ActualSales'].sum()/df['TargetSales'].sum())*100,2), "%")

# Charts
df.plot(x='Month', y=['TargetSales','ActualSales'], kind='bar')
plt.title("Target vs Actual Sales")
plt.show()

sns.lineplot(x='Month', y='Performance', data=df, marker='o')
plt.title("Performance Trend (%)")
plt.show()

# Forecast
X = np.arange(len(df)).reshape(-1,1)
y = df['ActualSales']
model = LinearRegression().fit(X, y)
next_month = np.array([[len(df)]])
print("Predicted Sales for May:", round(model.predict(next_month)[0],2))
