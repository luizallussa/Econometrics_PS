import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# 1. Load Data
data_dir = Path(r"C:\Users\luizallu\Downloads")
df = pd.read_csv(data_dir / "wage.csv")

# 2. Create log wage variable
df["lwage"] = np.log(df["wage"])

# Separate female (group 1) and male (group 2)
females = df[df["male"] == 0]["lwage"]
males = df[df["male"] == 1]["lwage"]

n1 = len(females)
n2 = len(males)

y1_bar = females.mean()
y2_bar = males.mean()

s1_sq = females.var(ddof=1)
s2_sq = males.var(ddof=1)

print(f"Females (n1 = {n1}): mean = {y1_bar:.4f}, s1^2 = {s1_sq:.6f}")
print(f"Males   (n2 = {n2}): mean = {y2_bar:.4f}, s2^2 = {s2_sq:.6f}")

# 3. Compute T statistic
T_obs = np.sqrt((n1 * n2) / (n1 + n2)) * (np.log(s1_sq) - np.log(s2_sq))
print(f"\nUnscaled T_obs: {T_obs:.4f}")

# 4. Compute kurtosis estimate (kappa_hat)
# kappa_hat = (n1 + n2) * sum_j sum_i (Y_ji - Y_j_bar)^4 / [sum_j sum_i (Y_ji - Y_j_bar)^2 ]^2
sum_fourth = np.sum((females - y1_bar) ** 4) + np.sum((males - y2_bar) ** 4)
sum_sq = np.sum((females - y1_bar) ** 2) + np.sum((males - y2_bar) ** 2)

kappa_hat = ((n1 + n2) * sum_fourth) / (sum_sq**2)
print(f"Estimated Kurtosis (kappa_hat): {kappa_hat:.4f}")

# 5. Robust Test Statistic
T_robust = T_obs / np.sqrt(kappa_hat - 1)
p_val_robust = 1 - stats.norm.cdf(T_robust)

# 6. Standard (Normal Theory) Test Statistic
T_standard = T_obs / np.sqrt(2)
p_val_standard = 1 - stats.norm.cdf(T_standard)


print("\n--- RESULTS ---")
print(
    f"1. Robust Test Statistic:   {T_robust:.4f}  (p-value = {p_val_robust:.4f})"
)
print(
    f"2. Standard Asymptotic Test: {T_standard:.4f}  (p-value = {p_val_standard:.4f})"
)

critical_value = stats.norm.ppf(0.95)
print(f"\nCritical Value at 5% level: {critical_value:.4f}")

if T_robust > critical_value:
    print("Robust Test: Reject H0")
else:
    print("Robust Test: Fail to Reject H0")

if T_standard > critical_value:
    print("Standard Test: Reject H0")
else:
    print("Standard Test: Fail to Reject H0")

"""
 ============
Females (n1 = 1569): mean = 1.4748, s1^2 = 0.397877
Males   (n2 = 1727): mean = 1.6930, s2^2 = 0.366413

Unscaled T_obs: 2.3621
Estimated Kurtosis (kappa_hat): 7.3781

--- RESULTS ---
1. Robust Test Statistic:   0.9353  (p-value = 0.1748)
2. Standard Asymptotic Test: 1.6703  (p-value = 0.0474)

Critical Value at 5% level: 1.6449
Robust Test: Fail to Reject H0
Standard Test: Reject H0
"""

