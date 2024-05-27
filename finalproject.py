#%%
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
file_path = './laptop_price.csv'
data = pd.read_csv(file_path)

# 변환 함수 정의
def convert_memory(memory):
    # 복합 저장소 처리
    if '+' in memory:
        parts = memory.split('+')
        total_memory = 0
        for part in parts:
            total_memory += convert_memory(part.strip())
        return total_memory
    
    # 단일 저장소 처리
    memory = memory.lower().replace('flash storage', '').replace('ssd', '').replace('hdd', '').strip()
    if 'tb' in memory:
        return int(re.sub(r'\D', '', memory)) * 1000  # 1TB = 1000GB
    elif 'gb' in memory:
        return int(re.sub(r'\D', '', memory))
    else:
        return 0

# 'Ram'과 'Memory' 변수에서 "GB"를 제거하고 숫자형으로 변환
data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)
data['Memory'] = data['Memory'].apply(convert_memory)

# 독립 변수(X)와 종속 변수(y) 설정
X_inch = data[['Inches']]
X_ram = data[['Ram']]
X_memory = data[['Memory']]
y = data['Price_euros']

# 각 변수와 Price_euros 간의 선형 회귀 모델 학습
model_inch = LinearRegression()
model_ram = LinearRegression()
model_memory = LinearRegression()

model_inch.fit(X_inch, y)
model_ram.fit(X_ram, y)
model_memory.fit(X_memory, y)

# 그래프 그리기
plt.figure(figsize=(15, 5))

# Inches와 Price_euros 관계 그래프
plt.subplot(1, 3, 1)
plt.scatter(X_inch, y, color='blue')
plt.plot(X_inch, model_inch.predict(X_inch), color='red')
plt.title('Inches vs Price')
plt.xlabel('Inches')
plt.ylabel('Price_euros')

plt.figure(figsize=(15, 5))

# Ram과 Price_euros 관계 그래프
plt.subplot(1, 3, 2)
plt.scatter(X_ram, y, color='green')
plt.plot(X_ram, model_ram.predict(X_ram), color='red')
plt.title('Ram vs Price')
plt.xlabel('Ram (GB)')
plt.ylabel('Price_euros')

plt.figure(figsize=(15, 5))

# Memory와 Price_euros 관계 그래프
plt.subplot(1, 3, 3)
plt.scatter(X_memory, y, color='orange')
plt.plot(X_memory, model_memory.predict(X_memory), color='red')
plt.title('Memory vs Price')
plt.xlabel('Memory (GB)')
plt.ylabel('Price_euros')

plt.tight_layout()
plt.show()

# 독립 변수(X)와 종속 변수(y) 설정
X = data[['Inches', 'Ram', 'Memory']]
y = data['Price_euros']

# 데이터셋을 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 평가 결과 출력
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 예측값 대 실제값 플롯
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# 잔차 플롯
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# %%
