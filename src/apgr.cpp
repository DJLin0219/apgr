#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

arma::vec grad_quad(arma::vec x, arma::mat A, arma::vec b){
  // Gradient of the squared error.
  //
  // Computes the gradient of the sum of squared error: \deqn{f(x) = 1/(2n)
  // \sum_{i=1}^n ( x'*A_i - b_i)^2}.
  //
  // @param x A p-dimensional vector where the gradient is computed.
  // @param opts List of parameters, which must include: \itemize{ \item \code{A},
  // a n*p design matrix, where each row is a sample and each column is a
  //   variable \item \code{b}, a n-dimensional response vector. }
  //
  // @return The gradient of the function \eqn{f(x) = 1/(2n) || Ax - b ||^2},
  //   which is \eqn{A'*(Ax - b)/n}.
  //
  // @export
  // @examples grad.quad(c(1,3,-2), A=diag(3), b=rep(1,3))
  
  int n = A.n_rows;
  arma::vec result = A.t()*(A*x - b)/n;
  return result;
}

arma::vec prox_l1(arma::vec x, double t, double lambda) {
  // Proximal operator of the scaled L1 norm.
  //
  // Computes the proximal operator of the L1 norm: h(x) = \lambda ||x||_1
  // ,} where \eqn{\lambda} is a scaling factor.
  //
  // @param x:The input vector
  // @param t:The step size
  // @param lambda:1-norm the scaling factor
  //
  // @return The proximal of \eqn{h} at {x} with step size \eqn{t}, given by
  //   \deqn{prox_h(x,t) = argmin_u [ t h(u) + 1/2 || x - u ||^2 ]}.
  //
  // @export
  // @examples prox.l1(x=c(1,3,-2), t=1.5, lambda=1)
  
  // Compute the soft-thresholded operator
  double thres = t * lambda;
  arma::uvec idx1 = find(x < -thres);
  arma::uvec idx2 = find(x > thres);
  arma::vec res = zeros<arma::vec>(x.size());
  if ( idx1.size()>0 ) res(idx1) = x(idx1) + thres;
  if ( idx2.size()>0 ) res(idx2) = x(idx2) - thres;
  return(res);
}


arma::vec prox_elasticnet(arma::vec x,double t,double lambda,double alpha){
  // Proximal operator of the scaled elastic net penalty.
  //
  // Computes the proximal operator of the scaled elastic net penalty: \deqn{h(x)
  // = \lambda [ (1 - \alpha)/2 ||x||_2^2 + \alpha ||x||_1 ] ,} where
  // \eqn{\lambda} is a scaling factor and \eqn{\alpha \in [0,1]} balances between
  // the L1 and L2 norms.
  //
  // @param x The input vector
  // @param t The step size (default is \code{1})
  // @param lambda : the scaling factor of the L1 norm 
  //        alpha: the scaling factor of the L2 norm 
  //
  // @return The proximal of \eqn{h} at {x} with step size \eqn{t}, given by
  //   \deqn{prox_h(x,t) = argmin_u [ t h(u) + 1/2 || x - u ||^2 ]}.
  //
  // @export
  // @examples prox.elasticnet(x=c(1,3,-2),t=1.5, lambda=1,alpha=0.5)
  
  // Compute the soft-thresholded operator
  double thres = t * lambda * alpha;
  arma::uvec idx1 = find(x < -thres);
  arma::uvec idx2 = find(x > thres);
  arma::vec res = zeros<arma::vec>(x.size());
  if ( idx1.size()>0 ) res(idx1) = x(idx1) + thres;
  if ( idx2.size()>0 ) res(idx2) = x(idx2) - thres;
  return(res / (1 + t * lambda * (1 - alpha)));
}

double gxfun(arma::vec x, arma::mat A, arma::vec b){
  double g_x;
  g_x = 1.0/2 * norm(A*x-b,2)*norm(A*x-b,2);
  return g_x;
}
// [[Rcpp::export]]
arma::vec apgr(arma::mat A, arma::vec b, double lambda, double gamma = 0,
               double t = 0.0055, double alpha = 1.01, double beta = 0.5, 
               int max_iteration = 2000, double eps = 1e-6, 
               int proximal= 2,int method=1,
               int step=1)
{
  // Accelerated proximal gradient optimization
  //
  // Notice: This package is the final assignment of statistics, ISBD department, Renmin University
  // of China.This R package is based on the apg R package written by Jean-Philippe and Vert.
  // We have expanded it to a certain extent.It isn't commercially available and 
  // we can delete at any time if infringement.
  //
  //
  
  // This function implements an accelerated proximal gradient method (Nesterov
  // 2007, Beck and Teboulle 2009). It solves: min_x (g(x) + h(x)), x in
  // R^dim_x, where g is smooth, convex and h  is non-smooth, convex
  // but simple so that we can easily evaluate the proximal operator of h.
  //
  // We suppose g(x)= norm(Ax-b)^2+h(x),norm is the 2-norm which g(x) is convex and smooth.
  // Parameter Input:
  //
  // @param A m*n matrix in g(x)
  // @param b m*1 vector in g(x)
  // @param lambda if h(x) is prox_l1 or prox_elsticnet, lambda is the coefficient of 1-norm
  //        h(x) = prox_l1(v,t,lambda) = lambda*t*1-norm(x)
  //        h(x) = prox_elsticnet(v,t,lambda,gamma) = t*(lambda*[(1-gamma)*1-norm(x)+gamma*2-norm(x)])
  // @param gamma the ratio of 1-norm and 2-norm
  // @param t the step-size of iteration, default=0.0039
  // @param alpha backtracking line search step-size growth factor, default 1.01
  // @param beta backtracking line search step-size shrinkage factor, default 0.5
  // @param max_iteration, the max iteration step, default 2000
  // @param eps, the thereshold of error, default 1e-6
  // @param proximal: 1 - prox_l1: lambda*||x||_1
  //                  2 - prox_elsticnet: lambda[(1 - gamma)/2 ||x||_2^2 + gamma ||x||_1 ]
  // @param method: 1- Proximal Gradient Descent
  //                2- Nesterov First Method 1987
  //                3- Nesterov Second Method 2007
  //                4- FISTA 
  // @param step:   1- Fixed step
  //                2- Barzilai-Borwein step-size
  //                3- Backtracking line-search
  //
  //Parameter output:
  // @param x: the argument minimize the target
  //
  //
  // @export
  // @examples # Solve a Lasso problem:
  // # min_x 1/2 norm( A%*%x - b )^2 + lambda ||x||_1
  // m <- 50
  // n <- 20
  // lambda <- 1
  // A <- matrix(rnorm(m*n), nrow=m)
  // b <- rnorm(m)
  // r <- apg(A,b,lambda,method=1,proximal=2,step=1)
  int dim_x =A.n_cols;                // The column number of A
  arma::vec x0 = zeros<arma::vec>(dim_x);         // Initial the x0   
  double theta = 1;
  double error = 0.0;
  
  arma::vec x = x0;                        // Initial the x0
  arma::vec xf = x0;                       // Initial the x-1
  arma::vec y = x0;                        // Initial the y
  arma::vec g = grad_quad(y,A,b);          // Initial the gradient
  
  
  //Main loop
  int i = 0;
  arma::vec v = x;
  for (i = 0; i < max_iteration; i++){
    
    //Record x and y
    xf = x;
    x = y;
    
    // Gradient method: PG/ACG1/ACG2
    if(method==1){     // PG
      v = x;
    }
    if(method==2){     //ACG1
      theta = double(i+1)/(i+4);
      v = x+(1-theta)*(x-xf);
    }
    if(method==3){     //ACG2
      theta = 2.0/(i+2);
      v = x+(1-theta)*(x-xf);
    }
    if(method==4){      //FISTA
      theta = 2/(1 + sqrt(1+4/(theta*theta)));
      v = x+(1-theta)*(x-xf);
    }
    
    // Calculate new gradient
    g = grad_quad(v,A,b);
    
    //Proximal Factor
    if(proximal == 1){
      y = prox_elasticnet(v-t*g, t, lambda, gamma);
    }
    else if(proximal == 2){
      y = prox_l1(v-t*g, t, lambda);
    }
    else if(proximal == 3){
      y = v-t*g;
    }
    
    // Calculate new gradient
    arma::vec gold = grad_quad(x,A,b); 
    g = grad_quad(y,A,b);
    
    //Calculate the error
    double x_norm = norm(x, 2);
    error = norm(y-x) / std::max(1.0, x_norm);
    
    if (error < eps) break;
    
    //step size
    if(step==1){      // B-B step
      double t_hat = 0.5*sum((y-x)%(y-x))/std::abs(sum((y - x)%(gold - g)));
      t = std::min(alpha*t, std::max(beta*t, t_hat));
      //      std::cout << t << "step = 1" << std::endl;
      
    }
    else if(step==2){  // fixed step
      t = 1*t;
    }
    else if(step == 3){  // backtracking step
      while (true){
        if(proximal == 1)
          y = prox_elasticnet(x-t*gold, t, lambda, gamma);
        else if(proximal == 2)
          y = prox_l1(x-t*gold, t, lambda);
        else if(proximal == 3)
          y = y-t*gold;
        
        if (gxfun(y,A,b) <= gxfun(x,A,b)+sum(A.n_cols*gold % (y-x))+ (1.0/(2*t))*sum((y-x)%(y-x)))
          break;
        
        t = beta*t;
      } 
    }
    
  }
  return x;
}
