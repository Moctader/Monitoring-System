import pandas as pd
import numpy as np

# Create a sample dataset
np.random.seed(42)
data = {
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.randint(20000, 120000, 1000),
    'purchased': np.random.choice([0, 1], 1000)
}

df = pd.DataFrame(data)
df.to_csv('customer_data.csv', index=False)
print("Sample CSV file 'customer_data.csv' created!")
