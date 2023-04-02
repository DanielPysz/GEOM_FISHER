import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from functools import partial
from sklearn.decomposition import TruncatedSVD

def climate_change(ideal_f, change):
    return ideal_f + change*np.ones(ideal_f.shape)
  
def meteorite_change(ideal_f, prob, change):
    if np.random.random() < prob:
        return ideal_f + change*np.ones(ideal_f.shape)
    return ideal_f
  
class Specimen:
    def __init__(self, features, mut_var, lamb, const=1e-3, mutations=set()):
        self.features = features
        self.mut_var = mut_var
        self.mutations = mutations
        self.lamb = lamb
        self.const = const

    def fitness(self, ideal_f, n_pop):
        return np.exp(-(np.linalg.norm(self.features-ideal_f)*self.const*n_pop))

    def child(self):
        mut = np.random.normal(0, self.mut_var, self.features.shape)
        child = Specimen(self.features, self.mut_var, self.lamb, mutations=self.mutations)
        child.features = child.features + mut
        child.mutations.add(np.linalg.norm(mut))
        return child

    def reproduce(self, ideal_f, n_pop):
        children = []
        if np.random.random() < self.fitness(ideal_f, n_pop):
            for i in range(np.random.poisson(self.lamb)):
                children.append(self.child())
        return children

    def add_trun_features(self, trunsvd):
        self.trun_features = trunsvd.transform(self.features.reshape(1, -1)).reshape(-1)
        
class Population:
    def __init__(self,
                 n_features: int,
                 f_init_var: float,
                 lamb: float,
                 n_init_pop: int,
                 mut_var_list: List[float],
                 change_func):
        self.population = [Specimen(# initial feature vector of the specimen
                                    np.random.multivariate_normal(np.zeros(n_features),
                                                                  np.identity(n_features)*(f_init_var)),
                                    # mut_var drawn from a distribution described by mut_var_init_prob
                                    np.random.choice(mut_var_list,
                                                     1)[0],
                                    lamb,
                                    ) for i in range(n_init_pop)]
        self.ideal_phen = np.zeros(n_features)
        self.change_func = change_func
        self.generations = [[self.population.copy(), self.ideal_phen.copy()]]

    def one_step(self):
        new_pop = []
        for specimen in self.population:
            new_pop += specimen.reproduce(self.ideal_phen, len(self.population))
        self.population = new_pop
        self.ideal_phen = self.change_func(self.ideal_phen)
        self.generations.append([self.population.copy(), self.ideal_phen.copy()])

    def simulation(self, num_steps):
        for _ in range(num_steps):
            self.one_step()

    def stack_one_pop(self, pop):
        stacked = np.stack([spec.features for spec in pop])
        return stacked


    def pca_on_features(self):
        stacked_features = np.concatenate([self.stack_one_pop(lst[0]) for lst in self.generations])

        trunsvd = TruncatedSVD(n_components=2)
        trunsvd.fit(stacked_features)

        self.generations_trun = self.generations.copy()
        for lst in self.generations_trun:
            for spec in lst[0]:
                spec.add_trun_features(trunsvd)
            lst[1] = trunsvd.transform(lst[1].reshape(1,-1)).reshape(-1)
        
    def one_population_frame(self, lst, time):
        pop, ideal = lst
        specimen_features = np.stack([spec.trun_features for spec in pop])

        df = pd.DataFrame(specimen_features, columns=['x','y'])
        df['time'] = time

        df['mut_group'] = [str(spec.mut_var) for spec in pop]
        df.loc[len(df),:] = {'x': ideal[0], 'y': ideal[1], 'time': time, 'mut_group': 'ideal'}
        return df

    def population_frames(self):
        return pd.concat([self.one_population_frame(lst, time) for time, lst in enumerate(self.generations_trun)]).reset_index(drop=True).reset_index()
    
pop = Population(2, 0.5, 1.2, 100, [0.05, 0.2, 0.5], partial(climate_change, change=0.00001))
pop.simulation(150)
pop.pca_on_features()
df = pop.population_frames()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

# create the figure and axis objects
fig, ax = plt.subplots()

# set the x and y axis limits
xlim = (min(df['x']) -0.5, max(df['x']) +0.5)
ylim = (min(df['y']) -0.5, max(df['y']) +0.5)

groups = np.unique(df['mut_group'])
num_groups = len(groups)
cmap = plt.get_cmap('gist_rainbow')
colors = {group: cmap(i / num_groups) for i, group in enumerate(groups)}

# define the function to update the scatter plot for each frame
def update(frame):
    # clear the previous plot elements
    ax.clear()
    # get the rows of the dataframe for the current frame
    data = df[df['time'] == frame]
    for i, group in enumerate(groups):
        ax.scatter(data[data['mut_group'] == group]['x'], data[data['mut_group'] == group]['y'], c=colors[group], label=group, alpha=0.5)
    ax.legend()
    ax.set_title('Frame {}'.format(frame))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

# create the animation object
ani = animation.FuncAnimation(fig, update, frames=df['time'].unique())

# display the animation as an HTML video
html = ani.to_jshtml()
HTML(html)

def plot_stacked_percentages(df):
    data = df[df['mut_group']!='ideal']
    grouped = data.groupby(['time', 'mut_group'])['x'].count()

    # Pivot the mut_group column into columns
    pivoted = grouped.unstack('mut_group')

    # Divide each value in the resulting dataframe by the sum of values in that row
    result = pivoted.div(pivoted.sum(axis=1), axis=0)
    result.fillna(0, inplace=True)

    plt.stackplot(range(int(max(result.index))+1), *[result[group] for group in result.columns], labels=result.columns)
    plt.legend(loc='upper left')
    plt.margins(0,0)
    plt.title('Area chart')
    plt.show()
    
    plot_stacked_percentages(df)



            
