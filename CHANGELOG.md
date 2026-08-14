# Changelog

## [Unreleased] - 2026-xx-xx

### Fixed
- Fixed `AttractorID` bookkeeping in Monte Carlo synchronous attractor detection in `BooleanNetwork.get_attractors_synchronous()`. Transient states encountered while discovering a new attractor are now correctly assigned to that attractor.
- Fixed double counting of newly discovered attractors in `BooleanNetwork.get_attractors_and_robustness_synchronous()`, which could cause `BasinSizesApproximation` to sum to greater than 1. `BasinSizesApproximation` may still sum to less than 1 when simulation timeouts occur.


## [1.0.2] - 2026-07-31

### Fixed
- Fixed integer overflow in `bin2dec` that could produce invalid results.


## [1.0.1] - 2026-06-12

### Added
- Added asynchronous dynamics and robustness computation, with terminal strongly connected components (SCCs) treated as attractors.


## [1.0.0] - 2026-03-25

### Added
- Initial release of BoolForge.