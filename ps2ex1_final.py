## PROBLEM SET 2
## 1a)

import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path(r"C:\Users\luizallu\Downloads")
df = pd.read_csv(data_dir / "ps1small.csv")

print(df.head())
print(df.shape)  # should be (898,3)

# create variable log wage
df["lwage"] = np.log(df["wage"])

# create dummy variables
# 6 education values (12-17) and 5 age values (26-30), so 30 dummies
# order: d1226, ..., d1230, d1326, ..., d1730

dnames = []
for e in range(12, 18):       
    for a in range(26, 31):  
        name = f"d{e}{a}"
        df[name] = ((df["education"] == e) & (df["age"] == a)).astype(int)
        dnames.append(name)

# regressor matrix: 898 x 30, each row is one person's d vector (transposed)
X = df[dnames]
y = df["lwage"]

# check
print(X.to_numpy().sum())   # should be 898


# Regression of Y on X

Xm = X.to_numpy()
ym = y.to_numpy()
n, k = Xm.shape                      # 898, 30

XtX_inv = np.linalg.inv(Xm.T @ Xm)   
alpha_hat = XtX_inv @ Xm.T @ ym           

resid = ym - Xm @ alpha_hat
sigma2 = resid @ resid / (n - k)     
se = np.sqrt(np.diag(sigma2 * XtX_inv))

# report results
results = pd.DataFrame({"coef": alpha_hat, "se": se}, index=dnames)
print(results)


"""
            coef        se
d1226   9.451320  0.090780
d1227   9.300426  0.084980
d1228   9.537856  0.083280
d1229   9.385826  0.085421
d1230   9.628323  0.087737
d1326   9.612120  0.192010
d1327   9.755901  0.192010
d1328   9.378158  0.174517
d1329   9.819982  0.223685
d1330   9.446488  0.202991
d1426   9.580658  0.223685
d1427   9.794423  0.170843
d1428   9.642237  0.216101
d1429   9.712199  0.174517
d1430   9.645440  0.170843
d1526   9.534555  0.241608
d1527   9.385619  0.232129
d1528   9.573155  0.264668
d1529   9.540361  0.252351
d1530   9.919473  0.241608
d1626  10.070500  0.167391
d1627   9.944426  0.187149
d1628   9.991987  0.147954
d1629  10.178524  0.130710
d1630  10.070591  0.164140
d1726   9.958317  0.483216
d1727  10.045760  0.223685
d1728  10.165648  0.295908
d1729   9.617564  0.418477
d1730  10.015981  0.374297


"""

## 1b)

from scipy.linalg import null_space
from scipy.stats import f

# G: 30 x 4, one row (1, e, a, e*a) per cell, same order as dnames
G = np.array([[1, e, a, e * a] for e in range(12, 18) for a in range(26, 31)])

# R: 26 x 30, rows orthogonal to the columns of G
R = null_space(G.T).T
p = R.shape[0]
print(R.shape)                 # (26, 30)
print(np.allclose(R @ G, 0))   # True: RG = 0

# F statistic (W/p with q = 0)
Ra = R @ alpha_hat
F_stat = Ra @ np.linalg.inv(sigma2 * R @ XtX_inv @ R.T) @ Ra / p

crit = f.ppf(0.95, p, n - k)
print(F_stat, crit)

# decision rule

if F_stat > crit:
    print(f"F = {F_stat:.3f} > {crit:.3f}: reject H0 at 5%")
else:
    print(f"F = {F_stat:.3f} <= {crit:.3f}: do not reject H0 at 5%")




