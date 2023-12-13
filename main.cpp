#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "kalman.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main()
{
  
  // std::cout << "Z.u: ";
  // for (double i: Z.u)
  //   std::cout << i << ' ';
  // std::cout << "\n" << std::endl;

  int n = 3; 
  int m = 1; 
  int p = 1;

  double dt = 1.0/30; // Time step

  MatrixXd A(n, n); 
  MatrixXd B(n, p);
  MatrixXd C(m, n);
  MatrixXd D(m, p); 
  MatrixXd Q(n, n); 
  MatrixXd R(m, m);
  MatrixXd S(n, m);
  MatrixXd Pf(n, n); 
  VectorXd xf(n);
  VectorXd u(p);
  VectorXd y(m);
  
  // Initial state
  xf = MatrixXd::Random(n,1);
  std::cout << "xf: \n" << xf << std::endl;

  // Some random measurements
  y = VectorXd::Random(m);
  u = VectorXd::Random(p);

  // Discrete LTI projectile motion, measuring position only
  A << 1, dt, 0, 0, 1, dt, 0, 0, 1;
  B << 0, 0.1, 0.1;
  C << 1, 0, 0;
  D << 0.5;

  // Reasonable covariance matrices
  Pf << .1, .1, .1, .1, 10000, 10, .1, 10, 100;
  Q << .05, .05, .0, .05, .05, .0, .0, .0, .0;
  R << 5;

  S = MatrixXd::Identity(n, m);

  Measurements Z(u, y);
  LtiModel M(A, B, C, D, Pf, Q, R, S, xf, Pf);

  Kalman rudi(Z, M);

  MatrixXd X = M.A;

  // Auxiliary matrices
  MatrixXd I_m(m, m);
  MatrixXd I_n(n, n);
  MatrixXd R_inv(m, m); 
  MatrixXd A_bar(n,n);
  MatrixXd B_bar(n, p);
  MatrixXd Q_bar(n, n);

  I_m = MatrixXd::Identity(m,m);
  I_n = MatrixXd::Identity(n,n);
  R_inv = R.ldlt().solve(I_m);  // ldlt or llt?  
  A_bar = A - S*R_inv*C;
  B_bar = B - S*R_inv*D;
  Q_bar = Q - S*R_inv*S.transpose();
  
  // Robust Cholesky?
  MatrixXd Pf_root(n, n);
  MatrixXd Q_root(m, m);
  Pf_root = Pf.llt().matrixL();
  Q_root = Q.llt().matrixL();

  // Block matrices
  MatrixXd QR(2*n, n);
  MatrixXd QR_Q(2*n, 2*n);
  MatrixXd QR_R(2*n, n);
  QR.block(0, 0, n, n) = Pf_root.transpose()*A_bar.transpose();
  QR.block(n, 0, n, n) = Q_root.transpose();
  
  Eigen::HouseholderQR<MatrixXd> qr(QR);
  QR_R = qr.matrixQR().triangularView<Eigen::Upper>();
  QR_Q = qr.householderQ();
  
  // XA=B --> A^T * X^T = B^T 
  MatrixXd K;
  MatrixXd BSR(n, m+p);
  VectorXd z(m+p);

  // BSR: the block matrix [B_bar S*R_inv]
  BSR.block(0, 0, n, p) = B_bar;
  BSR.block(0, p, n, m) = S*R_inv; 
  
  z.block(0, 0, p, 1) = u;
  z.block(p, 0, m, 1) = y;  
  
  K = (C*QR_R.transpose()*QR_R*C.transpose()+R).llt().solve(C*QR_R.transpose()*QR_R).transpose();
  xf = (I_n - K*C)*(A_bar*xf + BSR*z) + K*(y-D*u);  

  std::cout << "A: \n" << A << std::endl;
  std::cout << "C: \n" << C << std::endl;
  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "R: \n" << R << std::endl;
  std::cout << "P: \n" << Pf << std::endl;
  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "Inv: \n" << R*R_inv - I_m << std::endl;
  std::cout << "A_bar: \n" << A_bar << std::endl;
  std::cout << "P_root: \n" << Pf_root << std::endl;
  std::cout << "Q_root: \n" << Q_root << std::endl;
  std::cout << "QR: \n" << QR << std::endl;
  std::cout << "QR_R: \n" << QR_R << std::endl;
  std::cout << "QR_Q: \n" << QR_Q << std::endl;
  std::cout << "QR_QR: \n" << QR_Q*QR_R << std::endl;
  std::cout << "QR-QR_QR: \n" << QR-QR_Q*QR_R << std::endl;  
  std::cout << "K: \n" << K << std::endl;
  std::cout << "xf: \n" << xf << std::endl;
  
}
