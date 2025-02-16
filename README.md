# apgr: Accelerated Proximal Gradient Descent Optimization

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`apgr` is an R package that implements an accelerated proximal gradient descent optimization algorithm. It is designed to solve optimization problems of the form:

\[
\min_x (g(x) + h(x))
\]

where \( g(x) \) is a smooth, convex function, and \( h(x) \) is a non-smooth, convex function with an easily computable proximal operator.

This package was developed as part of an optimization course project at **Renmin University of China, Institute of Statistics and Big Data**.

---

## Installation

You can install the `apgr` package directly from GitHub using the `devtools` package:

```R
# Install devtools if you haven't already
install.packages("devtools")

# Install apgr from GitHub
devtools::install_github("DJLin0219/apgr")
