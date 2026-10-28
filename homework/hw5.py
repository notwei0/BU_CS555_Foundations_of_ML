import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#1
df = pd.read_csv("iq_groups.csv")

print(f"IQ: {df.groupby('group')['iq'].describe()}")
print("\n")
print(f"age: {df.groupby('group')['age'].describe()}")

axes = df.boxplot(column=["iq", "age"], by="group")
plt.suptitle("")
axes[0].set_title("IQ by group")
axes[0].set_ylabel("IQ")
axes[1].set_title("Age by group")
axes[1].set_ylabel("Age")
plt.tight_layout()
plt.savefig("boxplotiq.png")
plt.show()

#2
print("\n")
model = smf.ols("iq ~ group", data=df).fit()
anova_table = anova_lm(model)
print(anova_table)
print("\n")
tukey = pairwise_tukeyhsd(endog=df["iq"], groups=df["group"], alpha=0.05)
print(tukey.summary())

#3
print("\n")
df["is_math"] = (df["group"] == "Math student").astype(int)
df["is_physics"] = (df["group"] == "Physics student").astype(int)

model3 = smf.ols("iq ~ is_math + is_physics", data=df).fit()
print(model3.summary())

#4
print("\n")
mm = smf.ols("iq ~ age + C(group)", data=df).fit()
ancova_table = anova_lm(mm, typ=3)
print(ancova_table)
print(mm.summary())

#emmeans replacement
mean_age = df["age"].mean()

newdata = pd.DataFrame({
    "age": [mean_age, mean_age, mean_age],
    "group": ["Chemistry student", "Math student", "Physics student"]
})
newdata["group"] = newdata["group"].astype("category")
newdata["ls_mean_iq"] = mm.predict(newdata)
print("\n")
print(mean_age)
print(newdata)

#5
print("\n")
R2 = 0.90
k = 2
n = 25
F = (R2 * (n - k - 1)) / (k * (1 - R2))
print(F)