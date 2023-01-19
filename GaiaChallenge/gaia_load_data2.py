import pandas as pd
from datetime import datetime 

start_time = datetime.now() 
df = pd.read_csv(
    "GaiaChallenge/modelR1GaiaChallenge.csv",
    header=None
)
time_elapsed = datetime.now() - start_time 
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

df.columns = ["x", "y", "z", "vx","vy","vz"]
print(df)
data = df.values
#print(data[0])