import numpy as np
import pandas as pd
from pathlib import Path

### QUESTION 3d) ###
data_dir = Path("/Users/luizatavares/Downloads")
df = pd.read_excel(data_dir / "SP500Index.xlsx")
print(df.columns)
print(df.head())

# replace with the actual column name after checking the printout
sp = df["Level of the S&P 500 Index"].to_numpy()

# x_t = log(SP_t / SP_0); increments are the monthly log returns
dx = np.diff(np.log(sp))          # length T = len(sp) - 1
T = len(dx)

delta_ml = np.log(sp[-1] / sp[0]) / T      # = dx.mean()
sigma2_ml = np.mean((dx - delta_ml) ** 2)  # divisor T, not T-1
sigma_ml = np.sqrt(sigma2_ml)

print("T =", T)
print("delta_ML =", delta_ml)
print("sigma_ML =", sigma_ml)
print("annualized: drift =", 12 * delta_ml, ", vol =", np.sqrt(12) * sigma_ml)

