\name{apgr-package}
\alias{apgr-package}
\alias{apgr}
\docType{package}
\title{
\packageTitle{apgr}
}
\usage{
apgr(A,b,lambda,gamma,t,alpha,beta,max_iteration,eps,proximal,method,step)
}

\arguments{
\item{A}{m*n matrix in g(x)}

\item{b}{m*1 vector in g(x)}

\item{lambda}{if h(x) is prox_l1 or prox_elsticnet, lambda is the coefficient of 1-norm
       h(x) = prox_l1(v,t,lambda) = lambda*t*1-norm(x)
       h(x) = prox_elsticnet(v,t,lambda,gamma) = t*(lambda*[(1-gamma)*1-norm(x)+gamma*2-norm(x)])}

\item{gamma}{the ratio mixture of 1-norm and 2-norm}

\item{t}{the step-size of iteration, default=0.0039}

\item{alpha}{backtracking line search step-size growth factor, default 1.01}

\item{beta}{backtracking line search step-size shrinkage factor, default 0.5}

\item{max_iteration}{the max iteration step, default 2000}

\item{eps}{the thereshold of error, default 1e-6}

\item{proximal}{proximal iterator:\itemize{ 
\item \code{1}: prox_l1: h(x)=lambda*||x||_1
\item \code{2}: prox_elsticnet: h(x)=lambda[(1 - gamma)/2 ||x||_2^2 + gamma ||x||_1 ]
\item \code{3}: prox_0: h(x)=0}}

\item{method}{Acceleration Method:\itemize{
\item \code{1}:Proximal Gradient Descent
\item \code{2}:Nesterov First Method 1987
\item \code{3}:Nesterov Second Method 2007
\item \code{4}:FISTA 
}}

\item{step}{Step t selection:\itemize{
\item \code{1}:Fixed step
\item \code{2}:Barzilai-Borwein step-size
\item \code{3}:Backtracking line-search
}}
}

\description{
This function implements an accelerated proximal gradient method.
It solves: \deqn{min_x (g(x) + h(x)), x \in
R^dim_x} where \eqn{g} is smooth, convex and \eqn{h} is non-smooth, convex
but simple in that we can easily evaluate the proximal operator of h.
}
\value{
A list with \code{x}, the solution of the problem: \deqn{min_x (f(x)
  + h(x)), x \in R^dim_x ,}.
}
\examples{
# Solve a Lasso problem:
# min_x 1/2 norm( A\%*\%x - b )^2 + lambda ||x||_1
n <- 50
m <- 20
lambda <- 1
A <- matrix(rnorm(m*n), nrow=n)
b <- rnorm(n)
r <- apgr(A,b,lambda,method=1,proximal=2,step=2) 
}
