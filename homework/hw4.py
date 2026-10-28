import pandas as pd
import numpy as np
import scipy.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.plotting import scatter_matrix

#1
df = pd.read_csv("prestige.csv")
x = df["education"]
y = df["prestige"]

plt.scatter(x, y)
plt.xlabel("Education")
plt.ylabel("Prestige")
plt.title("Prestige vs Education")
plt.savefig("scatter.png")
plt.show()

education = df["education"]
prestige = df["prestige"]

np.corrcoef(education, prestige)[0,1]
np.corrcoef(prestige, education)[0,1]

print(np.corrcoef(education, prestige))

#2
model = smf.ols("prestige ~ education", data=df).fit()

m,b = np.polyfit(x, y, 1)
print("\n")
print(f"regression line: y = {m:.4f}x + {b:.4f}")

plt.scatter(x, y)
plt.plot(x, model.predict(df), color="red")
plt.xlabel("Education")
plt.ylabel("Prestige")
plt.title("Education vs Prestige")
plt.savefig("linear_reg1.png")
plt.show()

#residual plot
residuals = model.resid
fitted_vals = model.fittedvalues
income = df["income"]
women = df["women"]

plt.scatter(fitted_vals, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted prestige values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.savefig("resid1.png")
plt.show()

#to find outliers 
print("\n")
resids = pd.Series(model.resid.values, index=df["occupation"])
print(resids.sort_values().head(3))
print(resids.sort_values().tail(3))

#3
model3 = smf.ols("prestige ~ education + income + women", data=df).fit()
print(model3.summary())
print("\n")
print(stats.f.ppf(0.95, 3, 98))

plt.scatter(income, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("income")
plt.ylabel("residuals")
plt.title("income vs residuals")
plt.savefig("resid2.png")
plt.show()

plt.scatter(women, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("women")
plt.ylabel("Residuals")
plt.title("Residuals vs women")
plt.savefig("resid3.png")
plt.show()

print("\n")
b = model3.params
print(f"prestige-hat = {b['Intercept']:.3f} + {b['education']:.3f}*education + {b['income']:.5f}*income + {b['women']:.4f}*women")

#5
plt.scatter(model3.fittedvalues, model3.resid)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted prestige values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (all 3 predictors)")
plt.savefig("resid4.png")
plt.show()

resids3 = pd.Series(model3.resid.values, index=df["occupation"])
print("\n")
print(resids3.sort_values().head(3))
print(resids3.sort_values().tail(3))

#6
plt.scatter(df["education"], df["income"])
plt.xlabel("Education")
plt.ylabel("Income")
plt.title("Income vs Education")
plt.savefig("scatter2.png")
plt.show()

model6 = smf.ols("prestige ~ education * income", data=df).fit()
print("\n")
print(model6.summary())

plt.scatter(model6.fittedvalues, model6.resid)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted prestige values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (education * income)")
plt.savefig("resid5.png")
plt.show()

b1 = model6.params
print("\n")
print(f"prestige-hat = {b1['Intercept']:.3f} + {b1['education']:.3f}*education + {b1['income']:.5f}*income + {b1['education:income']:.5f}*education*income")