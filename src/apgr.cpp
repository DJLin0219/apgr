#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

//' Gradient of the squared error
//'
//' Computes the gradient of the sum of squared error: \eqn{f(x) = 1/(2n) || Ax - b ||^2}.
//'
//' @param x A p-dimensional vector where the gradient is computed.
//' @param A An n x p design matrix.
//' @param b An n-dimensional response vector.
//' @return The gradient of the function \eqn{f(x)}.
//' @export
arma::vec grad_quad(arma::vec x, arma::mat A, arma::vec b) {
  int n = A.n_rows;
  arma::vec result = A.t() * (A * x - b) / n;
  return result;
}

//' Proximal operator of the scaled L1 norm
//'
//' Computes the proximal operator of the L1 norm: \eqn{h(x) = \lambda ||x||_1}.
//'
//' @param x The input vector.
//' @param t The step size.
//' @param lambda The scaling factor.
//' @return The proximal of \eqn{h} at \eqn{x} with step size \eqn{t}.
//' @export
arma::vec prox_l1(arma::vec x, double t, double lambda) {
  double thres = t * lambda;
  arma::uvec idx1 = find(x < -thres);
  arma::uvec idx2 = find(x > thres);
  arma::vec res = zeros<arma::vec>(x.size());
  if (idx1.size() > 0) res(idx1) = x(idx1) + thres;
  if (idx2.size() > 0) res(idx2) = x(idx2) - thres;
  return res;
}

//' Proximal operator of the scaled elastic net penalty
//'
//' Computes the proximal operator of the scaled elastic net penalty: \eqn{h(x) = \lambda [ (1 - \alpha)/2 ||x||_2^2 + \alpha ||x||_1 ]}.
//'
//' @param x The input vector.
//' @param t The step size.
//' @param lambda The scaling factor of the L1 norm.
//' @param alpha The scaling factor of the L2 norm.
//' @return The proximal of \eqn{h} at \eqn{x} with step size \eqn{t}.
//' @export
arma::vec prox_elasticnet(arma::vec x, double t, double lambda, double alpha) {
  double thres = t * lambda * alpha;
  arma::uvec idx1 = find(x < -thres);
  arma::uvec idx2 = find(x > thres);
  arma::vec res = zeros<arma::vec>(x.size());
  if (idx1.size() > 0) res(idx1) = x(idx1) + thres;
  if (idx2.size() > 0) res(idx2) = x(idx2) - thres;
  return res / (1 + t * lambda * (1 - alpha));
}

double gxfun(arma::vec x, arma::mat A, arma::vec b) {
  double g_x;
  g_x = 1.0 / 2 * norm(A * x - b, 2) * norm(A * x - b, 2);
  return g_x;
}

//' Accelerated proximal gradient optimization
//'
//' This function implements an accelerated proximal gradient method (Nesterov 2007, Beck and Teboulle 2009).
//'
//' @param A An m x n matrix in the function \eqn{g(x)}.
//' @param b An m-dimensional vector in the function \eqn{g(x)}.
//' @param lambda The coefficient of the 1-norm in \eqn{h(x)}.
//' @param gamma The ratio of the 1-norm and 2-norm in \eqn{h(x)}.
//' @param t The step size of iteration (default is 0.0055).
//' @param alpha The backtracking line search step-size growth factor (default is 1.01).
//' @param beta The backtracking line search step-size shrinkage factor (default is 0.5).
//' @param max_iteration The maximum number of iterations (default is 2000).
//' @param eps The threshold of error (default is 1e-6).
//' @param proximal The proximal operator to use:
//'   \itemize{
//'     \item 1: prox_l1 (L1 norm)
//'     \item 2: prox_elasticnet (elastic net)
//'   }
//' @param method The acceleration method to use:
//'   \itemize{
//'     \item 1: Proximal Gradient Descent
//'     \item 2: Nesterov's First Method (1987)
//'     \item 3: Nesterov's Second Method (2007)
//'     \item 4: FISTA
//'   }
//' @param step The step size selection method:
//'   \itemize{
//'     \item 1: Fixed step
//'     \item 2: Barzilai-Borwein step-size
//'     \item 3: Backtracking line-search
//'   }
//' @return A vector \eqn{x} that minimizes the objective function.
//' @export
// [[Rcpp::export]]
arma::vec apgr(arma::mat A, arma::vec b, double lambda, double gamma = 0,
               double t = 0.0055, double alpha = 1.01, double beta = 0.5,
               int max_iteration = 2000, double eps = 1e-6,
               int proximal = 2, int method = 1, int step = 1) {
  int dim_x = A.n_cols;
  arma::vec x0 = zeros<arma::vec>(dim_x);
  double theta = 1;
  double error = 0.0;

  arma::vec x = x0;
  arma::vec xf = x0;
  arma::vec y = x0;
  arma::vec g = grad_quad(y, A, b);

  int i = 0;
  arma::vec v = x;
  for (i = 0; i < max_iteration; i++) {
    xf = x;
    x = y;

    if (method == 1) {
      v = x;
    } else if (method == 2) {
      theta = double(i + 1) / (i + 4);
      v = x + (1 - theta) * (x - xf);
    } else if (method == 3) {
      theta = 2.0 / (i + 2);
      v = x + (1 - theta) * (x - xf);
    } else if (method == 4) {
      theta = 2 / (1 + sqrt(1 + 4 / (theta * theta)));
      v = x + (1 - theta) * (x - xf);
    }

    g = grad_quad(v, A, b);

    if (proximal == 1) {
      y = prox_elasticnet(v - t * g, t, lambda, gamma);
    } else if (proximal == 2) {
      y = prox_l1(v - t * g, t, lambda);
    } else if (proximal == 3) {
      y = v - t * g;
    }

    arma::vec gold = grad_quad(x, A, b);
    g = grad_quad(y, A, b);

    double x_norm = norm(x, 2);
    error = norm(y - x) / std::max(1.0, x_norm);

    if (error < eps) break;

    if (step == 1) {
      double t_hat = 0.5 * sum((y - x) % (y - x)) / std::abs(sum((y - x) % (gold - g)));
      t = std::min(alpha * t, std::max(beta * t, t_hat));
    } else if (step == 2) {
      t = 1 * t;
    } else if (step == 3) {
      while (true) {
        if (proximal == 1) {
          y = prox_elasticnet(x - t * gold, t, lambda, gamma);
        } else if (proximal == 2) {
          y = prox_l1(x - t * gold, t, lambda);
        } else if (proximal == 3) {
          y = y - t * gold;
        }

        if (gxfun(y, A, b) <= gxfun(x, A, b) + sum(A.n_cols * gold % (y - x)) + (1.0 / (2 * t)) * sum((y - x) % (y - x))) {
          break;
        }

        t = beta * t;
      }
    }
  }
  return x;
}
