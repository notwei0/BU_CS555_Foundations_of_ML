#THIS IS FOR DATA CLEANING ONLY, NOT THE PROJECT ITSELF. I JUST WANTED TO STAY ORGANIZED!

import io
import urllib.request
import zipfile
import pandas as pd

url = "https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip"

with urllib.request.urlopen(url) as response:
    archive = zipfile.ZipFile(io.BytesIO(response.read()))

with archive.open("SouthGermanCredit.asc") as f:
    credit = pd.read_csv(f, sep=r"\s+")


# renames German column names to English (mapping published in the UCI variable documentation)
COLUMN_NAMES = {
    "laufkont": "status",              "laufzeit": "duration",
    "moral": "credit_history",         "verw": "purpose",
    "hoehe": "amount",                 "sparkont": "savings",
    "beszeit": "employment_duration",  "rate": "installment_rate",
    "famges": "personal_status_sex",   "buerge": "other_debtors",
    "wohnzeit": "present_residence",   "verm": "property",
    "alter": "age",                    "weitkred": "other_installment_plans",
    "wohn": "housing",                 "bishkred": "number_credits",
    "beruf": "job",                    "pers": "people_liable",
    "telef": "telephone",              "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}
df = credit.rename(columns=COLUMN_NAMES)

#verify the raw data before doing anything to it
print("data checks")
print("shape:", credit.shape)
print("missing values:", credit.isna().sum().sum())

#cleans the data to create scores of good and bad risk
df["bad_risk"] = 1 - df["credit_risk"]
print("\ncredit_risk (cleaned coding, 1 = bad, 0 = good):")
print(df["bad_risk"].value_counts().sort_index())

model_data = df[[
    "status",         "property", 
    "credit_history", "age", "bad_risk"
]].copy() #selected preditors

print("\nmodel data shape:", model_data.shape)

model_data.to_csv("cleaned_german_credit_data.csv", index=False) #converts into csv file
print("\ncleaned data saved as cleaned_german_credit_data.csv")