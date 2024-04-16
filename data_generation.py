# data_generation.py
import numpy as np
import pandas as pd

def generate_data(num_days=30):
    np.random.seed(42)
    timestamp = pd.date_range(start='20230401', periods=num_days*24, freq='H')
    load = np.random.normal(loc=50, scale=10, size=num_days*24)  # Simulate load on the bridge
    strain = load + np.random.normal(loc=0, scale=2, size=num_days*24)  # Simulated strain influenced by load
    acceleration = np.random.normal(loc=0, scale=1, size=num_days*24)  # Simulated vibrations/accelerations
    
    data = pd.DataFrame({
        'timestamp': timestamp,
        'load': load,
        'strain': strain,
        'acceleration': acceleration
    })
    return data
