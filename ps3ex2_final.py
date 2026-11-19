import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path(r"C:\Users\luizallu\Downloads")
# data_dir = Path("/Users/luizatavares/Downloads")   # use this line on the Mac
df = pd.read_excel(data_dir / "PS4data.xls")
print(df.columns)


### QUESTION 2a) ###
nondur = "real consumption of nondurables"
serv   = "real consumption of services"
pop    = "population"

# consumption per capita
df["C"] = (df[nondur] + df[serv]) / df[pop]

# lags 1 to 4
for l in range(1, 5):
    df[f"C_lag{l}"] = df["C"].shift(l)

# drop the first 4 rows, which have missing lags
d = df.dropna(subset=["C"] + [f"C_lag{l}" for l in range(1, 5)])

# y and X (constant first)
y = d["C"].to_numpy()
X = np.column_stack([np.ones(len(d))] + [d[f"C_lag{l}"].to_numpy() for l in range(1, 5)])

# OLS
n, k = X.shape
beta_hat = np.linalg.solve(X.T @ X, X.T @ y) 
resid = y - X @ beta_hat
RSS_u = resid @ resid

#print("n =", n, "k =", k)
#print("beta_hat =", beta_hat)
#print("RSS_u =", RSS_u)


# Test 3 hypotheses of R.beta=0

from scipy import stats

# R is 3x5: picks out beta2, beta3, beta4
R = np.array([[0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])
q = np.zeros(3)
p = R.shape[0]

XtX_inv = np.linalg.inv(X.T @ X)
sigma2_hat = RSS_u / (n - k)

# Wald statistic (homoskedastic)
diff = R @ beta_hat - q
W = diff @ np.linalg.inv(sigma2_hat * R @ XtX_inv @ R.T) @ diff

print("W =", W)
print("chi2(3) 5% critical value =", stats.chi2.ppf(0.95, p))
print("p-value (chi2) =", 1 - stats.chi2.cdf(W, p))

# F version, W/p
F = W / p
print("F =", F)
print("p-value (F) =", 1 - stats.f.cdf(F, p, n - k))




### QUESTION 2c) ###

nondur = "real consumption of nondurables"
serv   = "real consumption of services"
pop    = "population"

dinc = "real disposable income"

# consumption per capita
df["C"] = (df[nondur] + df[serv]) / df[pop]

# current income per capita
df["Y"] = df[dinc] / df[pop]

# log growth rates: log(C_t/C_t-1) and log(Y_t/Y_t-1)
df["dC"] = np.log(df["C"] / df["C"].shift(1))
df["dY"] = np.log(df["Y"] / df["Y"].shift(1))

# instruments: dC lagged 2 to 5, i.e. log(C_t-2/C_t-3), ..., log(C_t-5/C_t-6)
for l in range(2, 6):
    df[f"dC_lag{l}"] = df["dC"].shift(l)

# drop the first 6 rows, which have missing values
d = df.dropna(subset=["dC", "dY"] + [f"dC_lag{l}" for l in range(2, 6)])

# y, X and W (constant first)
y = d["dC"].to_numpy()
X = np.column_stack([np.ones(len(d)), d["dY"].to_numpy()])
W = np.column_stack([np.ones(len(d))] + [d[f"dC_lag{l}"].to_numpy() for l in range(2, 6)])

# Regressing X on W
X_hat = W @ np.linalg.solve(W.T @ W, W.T @ X)

# 2SLS
n, k = X.shape
gamma_hat = np.linalg.solve(X_hat.T @ X, X_hat.T @ y)
resid = y - X @ gamma_hat

print("n =", n, "k =", k, "l =", W.shape[1])
print("gamma_hat =", gamma_hat)


# White-robust variance of gamma_hat (Theorem 2 with V = sum e_i^2 w_i w_i')
C_hat = (W.T @ W) / n                 # (1/n) sum w_t w_t'
D_hat = (W.T @ X) / n                 # (1/n) sum w_t x_t'
V_hat = (W * resid[:, None]**2).T @ W / n     # (1/n) sum e_t^2 w_t w_t'

A_n = np.linalg.inv(D_hat.T @ np.linalg.inv(C_hat) @ D_hat) \
      @ D_hat.T @ np.linalg.inv(C_hat)        # k x l

var_gamma = (A_n @ V_hat @ A_n.T) / n         # overall 1/n here

se = np.sqrt(np.diag(var_gamma))
print("se =", se)
#print("t-stats =", gamma_hat / se)




### QUESTION 2d) ###

#part 1: endogeneity test (control function) 

# first stage: regress dY on W, keep residuals
pi_hat = np.linalg.solve(W.T @ W, W.T @ X[:, 1])   # X[:,1] is dY
u_hat = X[:, 1] - W @ pi_hat

# control function regression: dC on constant, dY, u_hat
Xc = np.column_stack([X, u_hat])                   # n x 3
delta_hat = np.linalg.solve(Xc.T @ Xc, Xc.T @ y)   # should be equal to the gammas 2sls
resid_c = y - Xc @ delta_hat

# Wald test of rho = 0 (coefficient on u_hat), robust version
XctXc_inv = np.linalg.inv(Xc.T @ Xc)
V_c = (Xc * resid_c[:, None]**2).T @ Xc
var_delta = XctXc_inv @ V_c @ XctXc_inv            # robust var of delta_hat

rho_hat = delta_hat[2]
se_rho = np.sqrt(var_delta[2, 2])
W_endog = (rho_hat / se_rho)**2

from scipy import stats
print("rho_hat =", rho_hat)
print("Wald endogeneity stat =", W_endog)
print("p-value =", 1 - stats.chi2.cdf(W_endog, 1))


#part 2: Sargan overidentification test

# S = n * e'W (W'W)^{-1} W'e / e'e, with e the 2SLS residuals (resid from 2c)
l_dim = W.shape[1]
S = n * (resid @ W @ np.linalg.solve(W.T @ W, W.T @ resid)) / (resid @ resid)

df_S = l_dim - k                   # 5 - 2 = 3
print("Sargan S =", S)
print("chi2(3) 5% critical value =", stats.chi2.ppf(0.95, df_S))
print("p-value =", 1 - stats.chi2.cdf(S, df_S))
