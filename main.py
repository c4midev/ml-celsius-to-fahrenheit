from sklearn.linear_model import LinearRegression
import pandas as pd
data = pd.read_csv("celsiusandfahrenheit.csv")

X = data["celsius"]
y = data["fahrenheit"]


processed_X = X.values.reshape(-1,1)
processed_y = y.values.reshape(-1,1)


model = LinearRegression()

model.fit(processed_X, processed_y)

while True:
    print("ML Celsius to Fahrenheit Model")
    print("--------------------------------")
    celsius_input = int(input("Celsius input data: "))
    calc = model.predict([[celsius_input]])
    print(f"Calculated fahrenheit: {calc}")