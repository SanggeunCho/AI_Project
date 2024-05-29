import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Get data from the file
file_path = './laptop_price.csv'
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

for encoding in encodings:
    try:
        data = pd.read_csv(file_path, encoding = encoding)
        print(f"Successfully read file. Used encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Encoding {encoding} failed")
else:
    print("All encoding attempts failed")

# Data preprocessing
def convert_memory(memory):
    # Remove the '+'
    if '+' in memory:
        parts = memory.split('+')
        total_memory = 0
        for part in parts:
            total_memory += convert_memory(part.strip())
        return total_memory
    
    # Remove unnecessary information
    memory = memory.lower().replace('flash storage', '').replace('ssd', '').replace('hdd', '').strip()
    if 'tb' in memory:
        return int(re.sub(r'\D', '', memory)) * 1000  # 1TB = 1000GB
    elif 'gb' in memory:
        return int(re.sub(r'\D', '', memory))
    else:
        return 0

# Leave only numbers
data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)
data['Memory'] = data['Memory'].apply(convert_memory)

# Apply exchange rates (1 Euro = 1.477(x 1,000Won))
exchange_rate = 1.477
data['Price_kr'] = data['Price_euros'] * exchange_rate

# Set the variables
X_inch = data[['Inches']]
X_ram = data[['Ram']]
X_memory = data[['Memory']]
y = data['Price_kr']

# Training
model_inch = LinearRegression()
model_ram = LinearRegression()
model_memory = LinearRegression()

model_inch.fit(X_inch, y)
model_ram.fit(X_ram, y)
model_memory.fit(X_memory, y)

# Graph of the Inches and Price relationship
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X_inch, y, color='blue')
plt.plot(X_inch, model_inch.predict(X_inch), color='red')
plt.title('Inches vs Price')
plt.xlabel('Inches')
plt.ylabel('Price (1000 KRW)')

# Graph of the Ram and Price relationship
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 2)
plt.scatter(X_ram, y, color='green')
plt.plot(X_ram, model_ram.predict(X_ram), color='red')
plt.title('Ram vs Price')
plt.xlabel('Ram (GB)')
plt.ylabel('Price (1000 KRW)')

# Graph of the Memory and Price relationship
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 3)
plt.scatter(X_memory, y, color='orange')
plt.plot(X_memory, model_memory.predict(X_memory), color='red')
plt.title('Memory vs Price')
plt.xlabel('Memory (GB)')
plt.ylabel('Price (1000 KRW)')

plt.tight_layout()
plt.show()

# Set the variables
X = data[['Inches', 'Ram', 'Memory']]
y = data['Price_kr']

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evalute the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mae}')
print(f'R^2 Score: {r2}')

# Plot of predicted values and actual values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices (1000 KRW)')
plt.ylabel('Predicted Prices (1000 KRW)')
plt.title('Actual vs Predicted Prices')
plt.show()

# Plot of residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Prices (1000 KRW)')
plt.ylabel('Residuals (1000 KRW)')
plt.title('Residuals vs Predicted Prices')
plt.axhline(0, color='red', linestyle='--')
plt.show()
