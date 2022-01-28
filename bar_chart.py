# import libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# set the style of the axes and the text color
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'

# create some fake data
emd = pd.Series([0.0312, 0.0301, 0.0285, 0.0318],
                        index=['RS', 'CEM', 'GD', 'RL'])
chamfer = pd.Series([0.0386, 0.0377, 0.0362, 0.0396],
                        index=['RS', 'CEM', 'GD', 'RL'])

df = pd.DataFrame({'emd' : emd, 'chamfer': chamfer})
df = df.sort_values(by='emd', ascending=False)

# we first need a numeric placeholder for the y axis
my_range=list(range(1,len(df.index)+1))

fig, ax = plt.subplots(figsize=(5,3.5))

# create for each expense type an horizontal line that starts at x = 0 with the length
# represented by the specific expense percentage value.
plt.hlines(y=my_range, xmin=0, xmax=df['emd'], color='#007ACC', alpha=0.4, linewidth=8)

# create for each expense type a dot at the level of the expense percentage value
plt.plot(df['emd'], my_range, "o", markersize=8, color='#007ACC', alpha=0.8, label='EMD')

plt.hlines(y=[i+0.2 for i in my_range], xmin=0, xmax=df['chamfer'], color='#CC5200', alpha=0.4, linewidth=8)

# create for each expense type a dot at the level of the expense percentage value
plt.plot(df['chamfer'], [i+0.2 for i in my_range], "o", markersize=8, color='#CC5200', alpha=0.8, label='CD')

# set labels
ax.set_xlabel('Distance', fontsize=15, fontweight='black', color = '#333F4B')
ax.set_ylabel('')

# set axis
ax.tick_params(axis='both', which='major', labelsize=12)
plt.yticks(my_range, df.index)
plt.legend(loc='upper right', markerscale=0.7, scatterpoints=1, fontsize=10)
# plt.legend(['CD'], loc='best', markerscale=0.7, scatterpoints=1, fontsize=10, labelcolor='#CC5200')
# add an horizonal label for the y axis
# fig.text(0, 0.96, '', fontsize=15, fontweight='black', color = '#333F4B')

# change the style of the axis spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_bounds((1, len(my_range)))
ax.set_xlim(0.025,0.04)

ax.spines['left'].set_position(('outward', 8))
ax.spines['bottom'].set_position(('outward', 5))

# plt.show()
plt.savefig('emd_chart.pdf', format='pdf', dpi=300, bbox_inches='tight')