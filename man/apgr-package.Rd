\name{apgr-package}
\alias{apgr-package}
\alias{apgr}
\docType{package}
\title{Accelerated Proximal Gradient Descent Optimization}

\usage{
apgr(A, b, lambda, gamma, t, alpha, beta, max_iteration, eps, proximal, method, step)
}

\arguments{
  \item{A}{
    An \(m \times n\) matrix representing \(A\) in \(g(x)\).
  }
  
  \item{b}{
    An \(m \times 1\) vector representing \(b\) in \(g(x)\).
  }
  
  \item{lambda}{
    If \(h(x)\) is \code{prox_l1} or \code{prox_elsticnet}, \code{lambda} is the coefficient of the 1-norm.
    \[
    h(x) = \text{prox_{l1}}(v, t, \lambda) = \lambda \cdot t \cdot ||x||_1
    \]
    \[
    h(x) = \text{prox_{elsticnet}}(v, t, \lambda, \gamma) = t \cdot \left(\lambda \cdot \left[(1-\gamma) \cdot ||x||_1 + \gamma \cdot ||x||_2^2\right]\right)
    \]
  }
  
  \item{gamma}{
    The ratio mixture of the 1-norm and 2-norm.
  }
  
  \item{t}{
    The step-size of iteration, default is \(0.0039\).
  }
  
  \item{alpha}{
    Backtracking line search step-size growth factor, default is \(1.01\).
  }
  
  \item{beta}{
    Backtracking line search step-size shrinkage factor, default is \(0.5\).
  }
  
  \item{max_iteration}{
    The maximum iteration steps, default is \(2000\).
  }
  
  \item{eps}{
    The threshold of error, default is \(1e-6\).
  }
  
  \item{proximal}{
    Proximal iterator:
    \begin{itemize}
      \item \code{1}: \code{prox_l1}: \(h(x) = \lambda \cdot ||x||_1\)
      \item \code{2}: \code{prox_elsticnet}: \(h(x) = \lambda \cdot \left[(1 - \gamma)/2 \cdot ||x||_2^2 + \gamma \cdot ||x||_1\right]\)
      \item \code{3}: \code{prox_0}: \(h(x) = 0\)
    \end{itemize}
  }
  
  \item{method}{
    Acceleration Method:
    \begin{itemize}
      \item \code{1}: Proximal Gradient Descent
      \item \code{2}: Nesterov First Method 1987
      \item \code{3}: Nesterov Second Method 2007
      \item \code{4}: FISTA
    \end{itemize}
  }
  
  \item{step}{
    Step \(t\) selection:
    \begin{itemize}
      \item \code{1}: Fixed step
      \item \code{2}: Barzilai-Borwein step-size
      \item \code{3}: Backtracking line-search
    \end{itemize}
  }
}

\description{
  This function implements an accelerated proximal gradient method for minimizing composite functions of the form \(g(x) + h(x)\), where \(g(x)\) is smooth, convex, and \(h(x)\) is non-smooth, convex, but simple in that we can easily evaluate its proximal operator.
}

\value{
  A list with \code{x}, the solution of the problem:
  \[
  \min_x (f(x) + h(x)), x \in \mathbb{R}^{dim_x}
  \]
}

\examples{
  # Solve a Lasso problem:
  # min_x 1/2 || A %*% x - b ||_2^2 + lambda ||x||_1
  n <- 50
  m <- 20
  lambda <- 1
  A <- matrix(rnorm(m * n), nrow = n)
  b <- rnorm(n)
  r <- apgr(A, b, lambda, method = 1, proximal = 2, step = 2)
}

