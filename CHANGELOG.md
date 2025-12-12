<!-- Types of changes

    Added for new features.
    Changed for changes in existing functionality.
    Deprecated for soon-to-be removed features.
    Removed for now removed features.
    Fixed for any bug fixes.
    Security in case of vulnerabilities. -->


# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.7.1] - 2025-08-29
### Fixed
- issue with speed bins in windroses


## [0.7.3] - 2025-09-04

### Removed
- removed depreciated pkg_resources dependency

### Added
- importlib_resources dependency

## [0.7.4] - 2025-09-25

### Changed
- replaced importlib_resources with importlib.resources

### Added
- added tools to read radiosonde met data downloaded files from Wyoming Weather Web

## [0.8.0] - 2025-12-12

Major fix of issues in the core solver, leading to more rapid convergence and faster execution.

### Notes
While moving towards release, this is still in development.
The API is incomplete and there are known bugs.

### Fixed
- more efficient implementation of boundary condition in Chebyshev pseudospectral solver

### Changed
- advanced requires-python to 3.11+
- new utilities to handle netcdf met
