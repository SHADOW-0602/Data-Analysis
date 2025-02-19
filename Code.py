import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv("ToyotaCorolla.csv")

print(df.head())
   
#Data Visualization

# 1. Price Distribution 
plt.figure(figsize=(8,5))
sns.histplot(df['Price'],bins=30,kde=True)
plt.title('Distribution of Toyota Corolla Prices')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

df['Age_08_04'].plot(kind='hist',bins=30,edgecolor='black',alpha=0.7)
plt.title("Age Distribution of Cars")
plt.xlabel("Age_08_04")
plt.show()

# 2.Correlation HeatMap
X = df[['Age_08_04', 'KM', 'Fuel_Type', 'HP', 'Met_Color', 'Automatic', 'cc', 'Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight']]
X = pd.get_dummies(X, drop_first=True)
df_corr=X.copy()

plt.figure(figsize=(12,6))
sns.heatmap(df_corr.corr(),annot=True,cmap='coolwarm',linewidth=0.5)
plt.title('Feature Correlation Map')
plt.show()

y = df['Price']

# 3. Scatter Plot: KM Driven vs. Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['KM'], y=df['Price'])
plt.title('KM Driven vs. Price')
plt.xlabel('KM Driven')
plt.ylabel('Price')
plt.show()

# 4. Price vs. Fuel Type
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Fuel_Type'], y=df['Price'])
plt.title('Fuel Type vs. Price')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.show()

# 5. Count Plot of Transmission Type (Automatic vs. Manual)
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Automatic'])
plt.title('Automatic vs. Manual Cars')
plt.xlabel('Transmission Type (0 = Manual, 1 = Automatic)')
plt.ylabel('Count')
plt.show()

# 6. Feature Importance using Correlation with Price
numeric_df = df.select_dtypes(include=[np.number])
correlation_with_price = numeric_df.corr()['Price'].sort_values(ascending=False)
print("\nTop Features Correlated with Price:\n", correlation_with_price)
