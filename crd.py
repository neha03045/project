import pandas
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
df = pandas.read_csv("cardata.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)
test_y = regr.predict(X)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
