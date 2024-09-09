import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# get the pyplot default color wheel
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def spider(df, *, id_column, title=None, subtitle=None, max_values=None, padding=1.25):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()

    lower_padding = (padding - 1)/2 
    # lower_padding = 0
    # upper_padding = 1 + lower_padding * 2
    upper_padding = 1 + 7 * lower_padding

    if max_values is None:
        max_values = {key: upper_padding*max(value) for key, value in data.items()}
        
    normalized_data = {key: np.array(value) / max_values[key] + lower_padding for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True), )
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        values = 1 - np.array(values) # Invert the values to have the higher values in the center
        ax.plot(angles, values, label=model_name,)
        ax.scatter(angles, values, facecolor=colors[i])
        ax.fill(angles, values, alpha=0.15)
        for _x, _y, t in zip(angles, values, actual_values):
            if _x == angles[2]:
                t = f'{t:.4f}' if isinstance(t, float) else str(t)
            else:
                t = f'{t:.2f}' if isinstance(t, float) else str(t)
            if t=='1': t = 'Model-free'
            if t=='2': t = 'Linear model'
            if t=='3': t = 'Nonlinear model'

            ax.text(_x, _y, t, size=10)
            
    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    if title is not None: plt.suptitle(title, fontsize=20)
    if subtitle is not None: plt.title(subtitle, fontsize=10)
    # plt.show()
    fig.savefig('result_2.png')
    
radar = spider

spider(
    pd.DataFrame({
        # 'x': [*'ab'],
        'x': ['GP-MPC', 'PPO'],
        '$\qquad\qquad\qquad\quad$ Performance\n': [3.94646538e-02, 0.03],
        # '$\quad\quad\quad\quad\quad\qquad$(Figure-8 tracking)': [3.94646538e-02, 0.03],
        'Generalization\nperformance\n': [0.024646868904967967, 0.1],
        'Inference\ntime\n\n': [0.016518109804624086, 0.0001351369751824273],
        '\n\n\nModel\ncomplexity ': [3, 1],
        '\nSampling\ncomplexity': [int(540), int(80*1e3)]
    }),
    id_column='x',
    title='   Overall Comparison',
    subtitle='(Normalized linear scale)',
    padding=1.1
)