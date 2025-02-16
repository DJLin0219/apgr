\name{apgr-package}
\alias{apgr-package}
\alias{apgr}
\docType{package}
\title{Accelerated Proximal Gradient Descent Optimization}
\usage{
apgr(A, b, lambda, gamma, t, alpha, beta, max_iteration, eps, proximal, method, step)
}

\arguments{
  \item{A}{An \eqn{m \times n} matrix in the function \eqn{g(x)}.}
  \item{b}{An \eqn{m \times 1} vector in the function \eqn{g(x)}.}
  \item{lambda}{The coefficient of the 1-norm in \eqn{h(x)}. If \eqn{h(x)} is \code{prox_l1} or \code{prox_elasticnet}, \code{lambda} is the coefficient of the 1-norm:
    \itemize{
      \item For \code{prox_l1}: \eqn{h(x) = \lambda \cdot t \cdot \|x\|_1}.
      \item For \code{prox_elasticnet}: \eqn{h(x) = t \cdot \lambda \cdot [\gamma \cdot \|x\|_1 + (1 - \gamma) \cdot \|x\|_2^2]}.
    }
  }
  \item{gamma}{The ratio mixture of the 1-norm and 2-norm in \code{prox_elasticnet}.}
  \item{t}{The step size of the iteration. Default is \code{0.0039}.}
  \item{alpha}{The step-size growth factor for backtracking line search. Default is \code{1.01}.}
  \item{beta}{The step-size shrinkage factor for backtracking line search. Default is \code{0.5}.}
  \item{max_iteration}{The maximum number of iterations. Default is \code{2000}.}
  \item{eps}{The error threshold for convergence. Default is \code{1e-6}.}
  \item{proximal}{The proximal operator to use:
    \describe{
      \item{\code{1}}{Proximal operator for L1 norm: \eqn{h(x) = \lambda \cdot \|x\|_1}.}
      \item{\code{2}}{Proximal operator for elastic net: \eqn{h(x) = \lambda \cdot [(1 - \gamma) \cdot \|x\|_1 + \gamma \cdot \|x\|_2^2]}.}
      \item{\code{3}}{Proximal operator for zero: \eqn{h(x) = 0}.}
    }
  }
  \item{method}{The acceleration method to use:
    \describe{
      \item{\code{1}}{Proximal Gradient Descent.}
      \item{\code{2}}{Nesterov's First Method (1987).}
      \item{\code{3}}{Nesterov's Second Method (2007).}
      \item{\code{4}}{FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).}
    }
  }
  \item{step}{The step size selection method:
    \describe{
      \item{\code{1}}{Fixed step size.}
      \item{\code{2}}{Barzilai-Borwein step size.}
      \item{\code{3}}{Backtracking line search.}
    }
  }
}

\description{
This function implements an accelerated proximal gradient method to solve optimization problems of the form:
\deqn{\min_x (g(x) + h(x)),}
where \eqn{g(x)} is a smooth, convex function and \eqn{h(x)} is a non-smooth, convex function with an easily computable proximal operator.
}

\value{
A list containing the following elements:
\item{x}{The solution vector.}
\item{iterations}{The number of iterations performed.}
\item{error}{The final error value.}
}

\examples{
# Solve a Lasso problem:
# min_x 1/2 * norm(A \%*\% x - b)^2 + lambda * (gamma * ||x||_1 + (1 - gamma) * ||x||_2)
n <- 50
m <- 20
lambda <- 1
A <- matrix(rnorm(m * n), nrow = n)
b <- rnorm(n)

# Run the accelerated proximal gradient method
result <- apgr(A = A, b = b, lambda = 0.01, gamma = 1, t = 0.0039,
               alpha = 1.01, beta = 0.5, max_iteration = 2000, eps = 1e-6,
               method = 3, proximal = 1, step = 1)

# Print the result
print(result)
}
