import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde

current_dir = os.getcwd()
file_names = ["RRF_1.json", "RRF_1.25.json", "LLMs.json", "CrossEncoder.json"]
files = [os.path.join(current_dir, f) for f in file_names]

data_mcc = []
data_bacc = []

for f in files:
    base_name = os.path.basename(f)
    label = os.path.splitext(base_name)[0]
    
    with open(f, 'r') as file:
        d = json.load(file)
        for item in d:
            if 'mcc' in item:
                data_mcc.append({'Label': label, 'mcc': item['mcc']})
            if 'balanced_accu' in item:
                data_bacc.append({'Label': label, 'balanced_accu': item['balanced_accu']})

df_mcc = pd.DataFrame(data_mcc)
df_bacc = pd.DataFrame(data_bacc)

# Helper function to plot bounded KDE
def plot_bounded_kde(df, column, title, xlabel, filename, lower_bound, upper_bound):
    plt.figure(figsize=(10, 6))
    
    # Create evaluation points strictly within the bounds
    x_eval = np.linspace(lower_bound, upper_bound, 500)
    
    labels = df['Label'].unique()
    for label in labels:
        # Extract data for this specific label
        data = df[df['Label'] == label][column].dropna().values
        
        # Ensure we have enough data points to compute KDE
        if len(data) > 1:
            kde = gaussian_kde(data)
            
            # Boundary Reflection Technique
            # 1. Standard density
            density = kde(x_eval)
            # 2. Reflect the left tail (spilling below lower_bound) back inside
            density += kde(2 * lower_bound - x_eval)
            # 3. Reflect the right tail (spilling above upper_bound) back inside
            density += kde(2 * upper_bound - x_eval)
            
            # Plot the line and fill the area under the curve
            line, = plt.plot(x_eval, density, label=label, linewidth=2)
            plt.fill_between(x_eval, density, alpha=0.3, color=line.get_color())

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.xlim(lower_bound, upper_bound)
    plt.legend(title='Label')
    plt.savefig(os.path.join(current_dir, filename))
    plt.close()

# Plot MCC (Bounded between -1 and 1)
plot_bounded_kde(
    df=df_mcc, 
    column='mcc', 
    title='Bounded KDE Plot of MCC Distributions', 
    xlabel='MCC', 
    filename='mcc_bounded_kdeplot.png', 
    lower_bound=-1.0, 
    upper_bound=1.0
)

# Plot Balanced Accuracy (Bounded between 0 and 1)
plot_bounded_kde(
    df=df_bacc, 
    column='balanced_accu', 
    title='Bounded KDE Plot of Balanced Accuracy Distributions', 
    xlabel='Balanced Accuracy', 
    filename='bacc_bounded_kdeplot.png', 
    lower_bound=0.0, 
    upper_bound=1.0
)