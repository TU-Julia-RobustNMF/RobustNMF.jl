# RobustNMF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/dev/)
[![Build Status](https://github.com/hai-sam/RobustNMF.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/hai-sam/RobustNMF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hai-sam/RobustNMF.jl)

This project implements **Non-negative Matrix Factorization (NMF)** methods in Julia, with a focus on robustness against noise and outliers.

The package provides:

-   a standard NMF implementation using an L2 (Frobenius) loss
-   utilities for synthetic data generation and preprocessing
-   tools to add noise and sparse outliers for robustness evaluation

---

## Features

-   **Standard NMF**

    -   Multiplicative update rules
    -   Frobenius (L2) reconstruction loss
    -   Convergence monitoring via error history

-   **Data preparation utilities**
    -   Synthetic non-negative data generation
    -   Gaussian noise injection
    -   Sparse outlier corruption
    -   Normalization to non-negative ranges
    -   Loading and preprocessing of image datasets

---
