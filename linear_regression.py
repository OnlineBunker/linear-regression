import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data2.csv')
X = df['studytime'].values
Y = df['score'].values

w = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X)

plt.scatter(X, Y)
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Scatter plot of Study Hours vs Exam Score')
plt.show()

error_array = []

def mean_squared_error(w, b, points):
    total_error = 0
    for x, y in points:
        total_error += (y - (w * x + b))**2
    return total_error / len(points)

for epoch in range(epochs):
    w_gradient = 0
    b_gradient = 0

    for x, y in list(zip(X, Y)):
        y_pred = w * x + b
        error = y - y_pred
        w_gradient += -2 * x * error
        b_gradient += -2 * error
    
    w -= (w_gradient / n) * learning_rate
    b -= (b_gradient / n) * learning_rate

    mse = mean_squared_error(w, b, list(zip(X, Y)))
    error_array.append(mse)
    print(f'Epoch {epoch+1}/{epochs}, w: {w:.4f}, b: {b:.4f}, MSE: {mse:.4f}')

plt.scatter(X, Y, label='Data')
predicted_Y = w * X + b
plt.plot(X, predicted_Y, color='red', label='Fitted Line')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

plt.plot(range(epochs), error_array, label='MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('MSE over Epochs')
plt.legend()
plt.show()