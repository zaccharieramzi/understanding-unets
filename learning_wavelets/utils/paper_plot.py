import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('science')
plt.rcParams['figure.figsize'] = (5.8, 2.5)
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


df_results = pd.read_csv('model_comparison.csv')

models_order = [
    # 'Wavelets',
    'Learnlet',
    'BM3D',
    'U-net 128',
]

model_legend = [
    r'\textbf{Learnlets}',
    'BM3d',
    'U-net',
]

colors = [f'C{i}' for i in range(len(models_order))]

reference = df_results['Wavelets']

cutoffs = [
    -75.5,
    -72,
    -3,
    -1.9,
    -0.3,
    2.5,
]

fig, (ax_normal, ax_outlier_1, ax_outlier_2, ax_legend) = plt.subplots(
    4, 
    1, 
    gridspec_kw=dict(
        height_ratios=[0.7, 0.1, 0.1, 0.1],
    ),
)
x_pos = np.arange(len(df_results))
x_names = [int(k) for k in df_results['std_noise']]
x_names[0] = r'$10^{-4}$'
for ax in [ax_normal, ax_outlier_1, ax_outlier_2]:
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_names)
    ax.minorticks_off()
handles = []
for i_model, model in enumerate(models_order):
    bars_model = df_results[model] - reference
    for i_ax, ax in enumerate([ax_normal, ax_outlier_1, ax_outlier_2]):
        b = ax.bar(x_pos + i_model * 0.12, bars_model, width=0.1, color=colors[i_model])
        if i_ax == 0:
            handles.append(b)
ax_normal.set_ylabel('PSNR difference with Wavelets \n (dB)')
ax_normal.yaxis.set_label_coords(0.09, 0.55, transform=fig.transFigure)
ax_normal.set_xlabel(r'$\sigma$')
ax_normal.xaxis.set_label_position('top') 

## Broken axis
ax_normal.set_ylim(*cutoffs[-2:])
ax_outlier_1.set_ylim(*cutoffs[2:4])
ax_outlier_2.set_ylim(*cutoffs[:2])
ax_normal.spines['bottom'].set_visible(False)
ax_outlier_1.spines['bottom'].set_visible(False)
ax_outlier_2.spines['top'].set_visible(False)
ax_outlier_1.spines['top'].set_visible(False)
ax_normal.xaxis.tick_top()
ax_normal.tick_params(labeltop=True)
ax_outlier_2.tick_params(labelbottom=False)
ax_outlier_2.xaxis.tick_bottom()
ax_outlier_1.set_xticks([])

## Cut-out diagonals
d = .008  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_normal.transAxes, color='k', clip_on=False, lw=0.5)
ax_normal.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_normal.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_outlier_2.transAxes)  # switch to the bottom axes
ax_outlier_2.plot((-d, +d), (1 - 5*d, 1 + 5*d), **kwargs)  # bottom-left diagonal
ax_outlier_2.plot((1 - d, 1 + d), (1 - 5*d, 1 + 5*d), **kwargs)  # bottom-right diagonal

kwargs.update(transform=ax_outlier_1.transAxes)  # switch to the bottom axes
ax_outlier_1.plot((-d, +d), (1 - 5*d, 1 + 5*d), **kwargs)  # bottom-left diagonal
ax_outlier_1.plot((1 - d, 1 + d), (1 - 5*d, 1 + 5*d), **kwargs)  # bottom-right diagonal
ax_outlier_1.plot((-d, +d), (-5*d, +5*d), **kwargs)        # top-left diagonal
ax_outlier_1.plot((1 - d, 1 + d), (-5*d, +5*d), **kwargs)  # top-right diagonal

## Legend
ax_legend.axis('off')
ax_legend.legend(handles, model_legend, loc='center', ncol=len(models_order),)

### Saving figure in high def
plt.savefig('model_comparison.pdf', dpi=600)