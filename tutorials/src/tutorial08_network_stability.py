# %% [markdown]
# # Perturbation and sensitivity analysis of Boolean networks
#
# In this tutorial, we study how Boolean networks respond to perturbations.
# Rather than implementing perturbations manually, we leverage BoolForge’s
# built-in robustness and sensitivity measures.
#
# ## What you will learn
# You will learn how to:
#
# - quantify robustness and fragility of Boolean networks under synchronous 
#   and asynchronous update schemes,
# - interpret basin-level and attractor-level robustness measures,
# - perform exact and approximate robustness computations, and
# - compute Derrida values as a measure of dynamical sensitivity.
#
# Together, these tools allow us to assess dynamical stability and resilience of 
# Boolean network models in a principled and computationally efficient way.
#
# ## Setup

# %%
import boolforge as bf
import pandas as pd

# %% [markdown]
# We reuse the small Boolean network from the previous tutorial as a running example.

# %%
string = """
x = y
y = x OR z
z = y
"""

bn = bf.BooleanNetwork.from_string(string, separator="=")

print("Variables:", bn.variables)
print("Number of nodes:", bn.N)

# %% [markdown]
# ## Exact attractors and robustness measures
#
# BoolForge provides a single method that computes:
#
# - all attractors,
# - basin sizes,
# - overall network coherence and fragility,
# - basin-level coherence and fragility, and
# - attractor-level coherence and fragility.
#
# These quantities are defined via systematic single-bit perturbations
# in the Boolean hypercube and can be computed *exactly* for small networks.

# %%
results_exact_sync = bn.get_attractors_and_robustness_synchronous_exact()
for key in results_exact_sync.keys():
    print(key)

# %% [markdown]
# For convenience, information about the dynamics (attractors, basin sizes, etc),
# described in detail in the previous tutorial, is also returned by this method.

# %%
print("Number of attractors:", results_exact_sync["NumberOfAttractors"])
print("Attractors (decimal states):", results_exact_sync["Attractors"])
print("Eventual attractor of each state:", results_exact_sync["AttractorID"])

print("Basin sizes:", results_exact_sync["BasinSizes"])

# %% [markdown]
# ## Network-, basin- and attractor-level robustness in synchronous networks
#
# Robustness can be resolved at different structural levels. Network-level metrics
# report the average robustness of any network state when subjected to perturbation. 

# %%
print("Overall coherence:", results_exact_sync["Coherence"])
print("Overall fragility:", results_exact_sync["Fragility"])

# %% [markdown]
# The same robustness metrics, coherence and fragility, can also be averaged
# across a smaller set of states, e.g., all states in one basin of attraction (see @bavisetty2025upper), or
# an even smaller set of states, e.g., all states that form an attractor (see @bavisetty2025attractors).


# %%
df_basins = pd.DataFrame({
    "BasinSizes": results_exact_sync["BasinSizes"],
    "BasinCoherences": results_exact_sync["BasinCoherences"],
    "BasinFragilities": results_exact_sync["BasinFragilities"],
}, index = list(map(str,results_exact_sync["Attractors"])))

df_attractors = pd.DataFrame({
    "AttractorCoherences": results_exact_sync["AttractorCoherences"],
    "AttractorFragilities": results_exact_sync["AttractorFragilities"],
}, index = list(map(str,results_exact_sync["Attractors"])))

print("Basin-level robustness:")
print(df_basins)

print("Attractor-level robustness:")
print(df_attractors)

# %% [markdown]
# Interpretation:
#
# - *Coherence* measures the fraction of single-bit perturbations that do *not*
#   change the final attractor [@willadsenwiles].
# - *Fragility*, only computed for synchronously updated networks, measures how much the
#    attractor state changes [@park2023models].
#
# The robustness metrics considered thus far describe how a single perturbation affects
# the network dynamics in the long-term, i.e., at the attractor. 
# These metrics are very meaningful biologically because attractors typically 
# correspond to cell types of phenotypes.
#
# It turns out that attractors in synchronously updated biological networks are 
# often less stable than their basins, a phenomenon explored in detail in Tutorial 10.

# %% [markdown]
# ## Network-, basin- and attractor-level robustness in asynchronous networks
# Coherence can also be defined for asynchronously updated networks, but fragility cannot, 
# because the notion of a single attractor state is not well-defined in the asynchronous case.
# `BoolForge` therefore only computes coherence for asynchronous networks.
# Similar to the synchronous case, `get_terminal_sccs_and_robustness_asynchronous_exact` returns
# attractor (terminal strongly connected component) information  and 
# information on robustness measures at the network, basin, and attractor levels.

# %%
results_exact_async = bn.get_terminal_sccs_and_robustness_asynchronous_exact()
print("Terminal strongly connected components (asynchronous attractors):", 
      results_exact_async["TerminalSCCs"])
print('Number of terminal strongly connected components:', 
      results_exact_async["NumberOfTerminalSCCs"])
print('Length of terminal strongly connected components:', 
      results_exact_async["LengthOfTerminalSCCs"])
print('Basin sizes (asynchronous):', 
      results_exact_async["BasinSizes"])

print("Overall coherence (asynchronous):", 
      results_exact_async["Coherence"])
print("Basin-level coherence (asynchronous):", 
      results_exact_async["BasinCoherences"])
print("Attractor-level coherence (asynchronous):", 
      results_exact_async["TerminalSCCCoherencesStationary"])

# %% [markdown]
# ## Approximate robustness for larger synchronous networks
#
# For larger networks, exact enumeration of all $2^N$ states is infeasible.
# BoolForge therefore provides a Monte Carlo approximation for synchronously updated networks
# that samples random initial conditions and perturbations. 

# %%
results_approx = bn.get_attractors_and_robustness_synchronous(n_simulations=500)

print("Number of attractors (lower bound):", results_approx["NumberOfAttractorsLowerBound"])
print("Approximate coherence:", results_approx["CoherenceApproximation"])
print("Approximate fragility:", results_approx["FragilityApproximation"])

# %% [markdown]
# Even when only using 500 random initial states, the approximate values closely match the exact ones.
# For networks with 20 to 30 nodes, simulation becomes the only feasible option due
# to the exponential growth of state space.
#
# Note that the approximate method is not yet implemented for asynchronously updated networks, 
# but this is planned for a future release, as the theory is being extended to the asynchronous case.

# %% [markdown]
# ## Derrida value: dynamical sensitivity
#
# An older and very popular robustness metric, the Derrida value, 
# measures how perturbations *propagate* after one synchronous update [@derrida1986random].
# It is defined as the expected Hamming distance between updated states that initially
# differed in exactly one bit. 
#
# BoolForge includes routines for the exact calculation and estimation of Derrida values.
# For networks with low degree, the exact calculation is strongly preferable. It is faster and more accurate.

# %%
derrida_exact = bn.get_derrida_value(exact=True)
derrida_approx = bn.get_derrida_value(n_simulations=2000)

print("Exact Derrida value:", derrida_exact)
print("Approximate Derrida value:", derrida_approx)

# %% [markdown]
# Interpretation:
#
# - Small Derrida values indicate ordered, stable dynamics.
# - Large Derrida values indicate sensitive or chaotic dynamics.
#
# Derrida values are closely related to average sensitivity of the update functions,
# and provide a complementary notion of robustness.

# %% [markdown]
# ## Summary
#
# In this tutorial you learned how to:
#
# - compute exact robustness measures for small Boolean networks,
# - interpret coherence and fragility at network, basin, and attractor levels,
# - approximate robustness measures for larger networks, and
# - assess dynamical sensitivity using the Derrida value.
#
# In Tutorial 9, we will finally analyze biological Boolean network models and
# design ensemble experiments. 
