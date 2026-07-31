#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:59:48 2026

@author: ckadelka
"""

import boolforge as bf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import time


def timer_generation(N,n,depth,exact_depth,allow_degenerate_functions=False):
    start = time.perf_counter()
    bf.random_network(N,n,depth,exact_depth,
                      allow_degenerate_functions=allow_degenerate_functions)
    end = time.perf_counter()
    return end-start


#number simulations
nsim=100

#network size parameters
N_min = 10
N_max = 20
N_step = 2

#constant degree parameter
ns = np.array([3,5,7])

#canalization/function parameters
depths = [0,100]
exact_depth=False    
allow_degenerate_functions = False


Ns = np.arange(N_min,N_max+1,N_step)

#warmup/initialization of numba, etc:
for N in Ns[:1]:
    for n in ns:
        for depth in depths:
            for _ in range(100):
                bf.random_network(N, n, depth, exact_depth=exact_depth, 
                                  allow_degenerate_functions=allow_degenerate_functions)

res2 = np.zeros((len(Ns),len(ns),len(depths),nsim))
for i,N in enumerate(Ns):
    for j,n in enumerate(ns):
        for k,depth in enumerate(depths):
            for ii in range(nsim):
                res2[i,j,k,ii] = timer_generation(N,n,depth,exact_depth,
                                                  allow_degenerate_functions)

cmap = matplotlib.cm.tab20
lss = ['-','--']
f,ax = plt.subplots(figsize=(4,3))
counter = 0
for j,n in enumerate(ns):
    for k,depth in enumerate(depths):
        data = 1000*res2[:,j,k,:]
        means = data.mean(1)
        ax.plot(Ns,means,color=cmap(counter),ls=lss[k])
        
        ses = data.std(axis=1,ddof=1) / np.sqrt(nsim)
    
        ax.fill_between(
            Ns,
            means - 1.96 * ses,
            means + 1.96 * ses,
            alpha=0.2,
            color=cmap(counter)
        )        

        counter+=1
        
ax.set_xlabel('Network size')
ax.set_ylabel('Network generation run time [ms]')

# ---------- legend 1: degree (color) ----------
degree_handles = [
    Line2D([0], [0],
           color=cmap(2*j),
           lw=2,
           label=fr'$n={n}$')
    for j, n in enumerate(ns)
]

legend1 = ax.legend(
    handles=degree_handles,
    title='Constant in-degree',
    loc='upper left',
    frameon=False
)

# ---------- legend 2: rule type (line style) ----------
style_handles = [
    Line2D([0], [0],
           color='black',
           lw=2,
           ls=lss[0],
           label='Random'),
    Line2D([0], [0],
           color='black',
           lw=2,
           ls=lss[1],
           label='Nested canalizing')
]

legend2 = ax.legend(
    handles=style_handles,
    title='Update rule ensemble',
    loc='center',
    ncol=2,
    bbox_to_anchor=[0.5,1.1],
    frameon=False
)

# keep first legend when adding second
ax.add_artist(legend1)
ax.spines[['top','right']].set_visible(False)
plt.savefig(f'runtime_generation_nsim{nsim}.pdf',bbox_inches='tight')











#time dynamics computation

def timer_evaluation(N,n,depth,exact_depth,
                     allow_degenerate_functions=False,
                     exact=True,
                     n_ICs_if_not_exact = 1000):
    bn = bf.random_network(N,n,depth,exact_depth)
    if exact:
        start = time.perf_counter()
        bn.get_attractors_and_robustness_synchronous_exact()
        end = time.perf_counter()
    else:
        start = time.perf_counter()
        bn.get_attractors_and_robustness_synchronous(n_simulations = n_ICs_if_not_exact)
        end = time.perf_counter()      
    return end-start

#number simulations
nsim=100
n_ICs_if_not_exact = 500

#network size parameters
N_min = 10
N_max = 20
N_step = 2

#constant degree parameter
n = 3

#canalization parameter
depths = [0,100]
exact_depth=False 
allow_degenerate_functions=False

#computation modes
exacts = [False, True]   

Ns = np.arange(N_min,N_max+1,N_step)   

#warmup/initialization of numba, etc:
for N in Ns[:1]:
    for n in ns:
        for depth in depths:
            for _ in range(100):
                bf.random_network(N, n, depth, exact_depth=exact_depth, 
                                  allow_degenerate_functions=allow_degenerate_functions)


res = np.zeros((len(Ns),len(ns),len(depths),nsim))
for i,N in enumerate(Ns):
    for j,exact in enumerate(exacts):
        for k,depth in enumerate(depths):
            for ii in range(nsim):
                out = timer_evaluation(N,n,depth,exact_depth,
                                       allow_degenerate_functions=allow_degenerate_functions,
                                       exact=exact,
                                       n_ICs_if_not_exact=n_ICs_if_not_exact)
                res[i,j,k,ii] = out
            print(N,exact,k)
                
cmap = matplotlib.cm.tab20
counter_min=6
lss = ['-','--']
f,ax = plt.subplots(figsize=(4,3))
counter = counter_min
for j,exact in enumerate(exacts):
    for k,depth in enumerate(depths):
        data = res[:,j,k,:]
        means = data.mean(1)
        ax.semilogy(Ns,means,color=cmap(counter),ls=lss[k])
        
        ses = data.std(axis=1,ddof=1) / np.sqrt(nsim)
    
        ax.fill_between(
            Ns,
            (means - 1.96 * ses),
            (means + 1.96 * ses),
            alpha=0.2,
            color=cmap(counter)
        )        

        counter+=1
        
ax.set_xlabel('Network size')
ax.set_ylabel('Dynamical analysis run time [s]')

# ---------- legend 1: degree (color) ----------
degree_handles = [
    Line2D([0], [0],
           color=cmap(counter_min+2*j),
           lw=2,
           label='Approximation' if not exact else 'Exact computation',
           ) for j,exact in enumerate(exacts)
]

legend1 = ax.legend(
    handles=degree_handles,
    title='Method',
    loc='lower right',
    frameon=False
)

# ---------- legend 2: rule type (line style) ----------
style_handles = [
    Line2D([0], [0],
           color='black',
           lw=2,
           ls=lss[0],
           label='Random'),
    Line2D([0], [0],
           color='black',
           lw=2,
           ls=lss[1],
           label='Nested canalizing')
]

legend2 = ax.legend(
    handles=style_handles,
    title='Update rule ensemble',
    loc='center',
    ncol=2,
    bbox_to_anchor=[0.5,1.1],
    frameon=False
)

# keep first legend when adding second
ax.add_artist(legend1)
[y1,y2] = ax.get_ylim()
ax.set_ylim([y1,y2])
ax.spines[['top','right']].set_visible(False)
plt.savefig(f'runtime_dynamics_nsim{nsim}_n{n}.pdf',bbox_inches='tight')












