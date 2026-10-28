import pandas as pd               
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

# 1 uploads the data
df = pd.read_csv("hospital.csv")  

#2 draw the histogram
plt.hist(df["days"], bins=range(1, 17), edgecolor="black")
plt.xlabel("Duration of stay")
plt.ylabel("Number of patients")
plt.title("Hospital Stay Duration for C. Difficile Patients")
plt.savefig("histogram.png")
plt.show()

#3 shows more data like median and etc
print("\n")
print("3")
print(df["days"].describe())

#4 finds the z score
print("\n")
print("4")
z = (df["days"] - df["days"].mean()) / df["days"].std()
print(pd.DataFrame(z.values.reshape(10, 10)).round(2).to_string(index=False, header=False))

#5 deterines probabilities when new mean is 5 and new std is 3
#a 
print("\n")
print("5a")
p1 = stats.norm.cdf(10, 5, 3) - stats.norm.cdf(3, 5, 3)
p = stats.norm.cdf(10, 5, 3)
print(f"{p:.1%}")
print(f"{p1:.1%}")

#b
print("\n")
print("5b")
samplemean = 3 / math.sqrt(35)
p2 = 1 - stats.norm.cdf(6, 5, samplemean)
print(f"{p2:.1%}")

#6 introducing heights and using a new z-score for the new data
heights =  [156, 156, 157, 167, 168, 168, 169, 171,
            171.5, 172, 173, 174, 176, 177, 177, 178, 
            178, 178, 181, 182, 183, 183, 184, 184, 
            185, 186, 187, 188, 189, 189 , 189.5]

#a
print("\n")
print("6a")
height_z = (heights - pd.Series(heights).mean()) / pd.Series(heights).std()
print(height_z.round(2))

#b
print("\n")
print("6b")
p2 = 1 - stats.norm.cdf(190, pd.Series(heights).mean(), pd.Series(heights).std())
print(f"{p2:.1%}")

#c what is the probability of 190 or higher being the average
print("\n")
print("6c")
samplemean1 = pd.Series(heights).std() / math.sqrt(len(heights))
p3 = 1 - stats.norm.cdf(190, pd.Series(heights).mean(), samplemean1)
print(f"{p3}")