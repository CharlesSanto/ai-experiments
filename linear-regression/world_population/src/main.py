import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

df = pd.read_csv("../data/world_population.csv")

years = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]

world_population = []

for year in years:
    column = f"{year} Population"
    total_population = df[column].sum()

    world_population.append(total_population)

X = np.array(years).reshape(-1, 1)  
y = np.array(world_population)

model = LinearRegression()
model.fit(X, y)

future_years = np.array([2023, 2024, 2025, 2026, 2027]).reshape(-1, 1)

predictions = model.predict(future_years)

all_years = np.concatenate((years, future_years.flatten()))
all_predictions = model.predict(all_years.reshape(-1, 1))

plt.figure(figsize=(12, 6))

plt.scatter(
    years,
    world_population,
    s=100,
    label="Dados reais"
)

plt.plot(
    all_years,
    all_predictions,
    linewidth=3,
    label="Regressão Linear"
)

plt.scatter(
    future_years,
    predictions,
    s=100,
    marker="x",
    label="Previsões futuras"
)

plt.xlabel("Ano", fontsize=12)
plt.ylabel("População Mundial", fontsize=12)

plt.title(
    "Previsão da População Mundial com Regressão Linear",
    fontsize=16
)

plt.grid(True)

plt.legend()

plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()

plt.show()