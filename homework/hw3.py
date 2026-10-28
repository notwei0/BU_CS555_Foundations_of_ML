import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import numpy as np

#1 creates a scatterplot for the data
df = pd.read_csv("fish.csv")
x = df["fish_meals"]
y = df["mercury"]

plt.scatter(x, y)
plt.xlabel("Number of fish meals per week")
plt.ylabel("Total mercury in hair in mg/g")
plt.title("Mercury Level vs Fish Meals per Week")
#plt.savefig("scatter.png")
#plt.show()

#2 calculates the correlation coefficient 
fish_meals = df["fish_meals"]
mercury = df["mercury"]

np.corrcoef(fish_meals, mercury)[0,1]
np.corrcoef(mercury, fish_meals)[0,1]

print(np.corrcoef(fish_meals, mercury))

#3 calculates the least squares regression line

model = smf.ols("mercury ~ fish_meals", data=df).fit()
conf_int = model.conf_int(alpha=0.1)
print(model.summary())


#prints the equation of the regression line
m,b = np.polyfit(x, y, 1)
print("\n")
print(f"regression line: y = {m:.4f}x + {b:.4f}")

#adds the regression line to the scatterplot
plt.scatter(x, y)
plt.plot(x, model.predict(df), color="red")
plt.xlabel("Number of fish meals per week")
plt.ylabel("Total mercury in hair in mg/g")
plt.title("Mercury Level vs Fish Meals per Week")
plt.savefig("linear_reg.png")
plt.show()

#5
print("\n")

df3 = {"fish_meals": fish_meals, "mercury": mercury}
model = smf.ols("mercury ~ fish_meals", data=df3).fit()

anova_table = anova_lm(model)
print(anova_table)

R_square = anova_table.loc["fish_meals", "sum_sq"] / anova_table["sum_sq"].sum()
print(R_square)

model = smf.ols("mercury ~ fish_meals", data=df3).fit()
anova_table = anova_lm(model)
print(anova_table)

print("\n")
print(conf_int)
print(f"{stats.t.ppf(1 - 0.05/2, 98):.4f}")