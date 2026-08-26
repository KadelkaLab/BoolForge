# Changelog

## [1.0.3] - 2026-xx-xx

### Added (major functionality)
- added support for computing exact dynamics and robustness for Boolean networks updated under the SDDS framework. Idea: Asynchronous functionality is a special case of a stochastic update. General asynchronous update and SDDS update just differs in the way the transition matrix is computed. Once computed, all downstream analyses are the same. Therefore, a number of methods containing the word `stochastic` have been added that take as first input the keyword `update_scheme`, which can be `asynchronous` or `sdds`. If `sdds`, activation and degradation probabilities for each node need to be also provided.

### Added (minor functionality)
- `get_attractors_and_robustness_synchronous` has now an optional argument `return_attractorID`. If True (default False), the mapping of sampled states to attractors is returned. This is useful for pairwise comparisons between base and controlled networks, plus it is something `get_attractors_synchronous` was already doing anyways. 

### Fixed
- Fixed `AttractorID` bookkeeping in Monte Carlo synchronous attractor detection in `.get_attractors_synchronous()`. Transient states encountered while discovering a new attractor are now correctly assigned to that attractor.
- Fixed double counting of newly discovered attractors in `.get_attractors_and_robustness_synchronous()`, which could cause `BasinSizesApproximation` to sum to greater than 1. `BasinSizesApproximation` may still sum to less than 1 when simulation timeouts occur.
- Fixed an error in `.summary(compute_all=True)` when dynamics are computed by simulation for networks of size N>=15. 

## [1.0.2] - 2026-07-31

### Fixed
- Fixed integer overflow in `bin2dec` that could produce invalid results.


## [1.0.1] - 2026-06-12

### Added
- Added asynchronous dynamics and robustness computation, with terminal strongly connected components (SCCs) treated as attractors.


## [1.0.0] - 2026-03-25

### Added
- Initial release of BoolForge.