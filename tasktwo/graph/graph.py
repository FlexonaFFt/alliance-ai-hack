import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results.csv')
plt.figure(figsize=(14, 7))
plt.plot(
    data['iteration'],
    data['ml_score'],
    marker='o',
    linestyle='-',
    color='b',
    linewidth=2,
    markersize=8
)

plt.title('График скора по попыткам', fontsize=16, pad=20)
plt.xlabel('Попытка', fontsize=14)
plt.ylabel('Скор', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

for i, row in data.iterrows():
    plt.annotate(
        f"{row['ml_score']:.4f}",
        (row['iteration'], row['ml_score']),
        textcoords="offset points",
        xytext=(0, 15),
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, edgecolor='gray')
    )

plt.tight_layout()
plt.savefig('improved_results_plot.png', dpi=300, bbox_inches='tight')
plt.show()