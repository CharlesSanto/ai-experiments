import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("student_scores_large.csv")

x = data[["hours_studied"]]
y = data["score"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

x_line = np.linspace(x.min().iloc[0], x.max().iloc[0], 100).reshape(-1, 1)

y_linear = linear_model.predict(x_line)

# score
y_linear_test = linear_model.predict(x_test)
linear_score = r2_score(y_test, y_linear_test)

poly = PolynomialFeatures(degree=2)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

polynomial_model = LinearRegression()
polynomial_model.fit(x_train_poly, y_train)

x_line_poly = poly.transform(x_line)
y_polynomial = polynomial_model.predict(x_line_poly)

# score
y_poly_test = polynomial_model.predict(x_test_poly)
poly_score = r2_score(y_test, y_poly_test)

prediction = polynomial_model.predict(
    poly.transform([[9]])
)

print(f"Prediction for 9 hours: {prediction[0]:.2f}")

plt.scatter(x, y, label="Real Data")

plt.plot(
    x_line,
    y_linear,
    label=f"Linear Regression (R²={linear_score:.2f})"
)

plt.plot(
    x_line,
    y_polynomial,
    label=f"Polynomial Regression (R²={poly_score:.2f})"
)

plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()

plt.show()