import pandas as pd

df = pd.DataFrame({'a':[1,2], 'b':[(1,2), (3,4)]})

print(type(df['b'][0]))
