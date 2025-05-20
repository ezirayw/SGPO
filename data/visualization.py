import pandas  as pd
import logomaker
import numpy as np

def process_logo(batch, sites=None):
  if sites == None:
    sites = np.arange(1, len(batch["sequence"][0]) + 1)
  batch["combo"] = batch["sequence"].apply(lambda x: ''.join([x[i-1] for i in sites]))
  #df["combo"] = df["sequence"].apply(lambda x: x[182] + x[183] + x[226] + x[227])

  dfs = []
  for i in range(len(sites)):
    batch[f'AA{i}'] = batch['combo'].str[i]
    temp = batch.groupby(f'AA{i}').count()
    temp[i] = temp.mean(axis=1)
    temp = temp[[i]]
    temp.index.name = ''
    temp = temp.copy().T
    dfs.append(temp)
  
  df = pd.concat(dfs)
  #replace nan with 0
  df = df.fillna(0)
  #normalize rows to 1
  df = df.div(df.sum(axis=1), axis=0)
  return df

def plot_logo(df, name, ax, sites):
  # create Logo object
  AAs_logo = logomaker.Logo(df,
                            color_scheme='weblogo_protein',
                            font_name='Arial',
                            ax=ax,
                            vpad=.15,
                            # figsize=(0.5*len(sites), 3),
                          #   baseline_width=.8,
  )

  # additional styling using Logo methods
  AAs_logo.style_spines(visible=False)

  # style using Axes methods
  AAs_logo.ax.set_ylim([0, 1])
  #AAs_logo.ax.set_ylabel('frequency', labelpad=0)
  AAs_logo.ax.set_yticks([])
  #AAs_logo.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
  AAs_logo.ax.set_xticks(range(len(sites)))
  AAs_logo.ax.set_xticklabels(sites)
  AAs_logo.ax.set_title(name)
  return AAs_logo