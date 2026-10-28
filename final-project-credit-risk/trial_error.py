import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("cleaned_german_credit_data.csv")

# fit a multiple linear regression model using statsmodels using ols and not glm
model = smf.ols("bad_risk ~ status + property + credit_history + age", data=df).fit()
print(model.summary())

#plots each predictor against bad risk
#after seeing the summary of both i was curious to see if the predictors were linear with the response variable, so i plotted each predictor against bad risk to see if there was a linear relationship
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

#plots residuals vs fitted values to check for linearity and homoscedasticity, in which it produced 2 lines due to the categorical variables, but the residuals are both in 2 diagnostic lines, which is not ideal, but it is still a good check to see if the model is linear and homoscedastic
plt.scatter(model.fittedvalues, model.resid_response)
plt.axhline(0, linestyle = "--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()


for ax, var, title in [
    (axes[0, 0], "status",         "bad-risk rate by checking account status"),
    (axes[0, 1], "property",       "bad-risk rate by property"),
    (axes[1, 0], "credit_history", "bad-risk rate by credit history"),
]:
    rate = df.groupby(var)["bad_risk"].mean()
    ax.bar(rate.index.astype(str), rate.values)
    ax.axhline(df["bad_risk"].mean(), linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel(var)
    ax.set_ylabel("proportion bad")
    ax.set_ylim(0, 0.7)

ax = axes[1, 1]
ax.boxplot([df.loc[df["bad_risk"] == 0, "age"],
            df.loc[df["bad_risk"] == 1, "age"]])
ax.set_xticks([1, 2])
ax.set_xticklabels(["good (0)", "bad (1)"])
ax.set_title("Age by credit outcome")
ax.set_ylabel("age (years)")

plt.tight_layout()
plt.show()
