#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "kalman.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main()
{
  int n = 3; 
  int m = 1; 
  int p = 1;
  int N = 20;

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
  MatrixXd U;
  MatrixXd Y;
  
  // Initial state
  xf = MatrixXd::Random(n,1);
  std::cout << "xf: \n" << xf << std::endl;

  // Some random measurements
  U = MatrixXd::Random(p, N);
  Y = MatrixXd::Random(m, N);
  
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

  Measurements Z(U, Y);
  LtiModel M(A, B, C, D, Pf, Q, R, S, xf, Pf);

  std::vector<VectorXd> gainSequence;
  std::vector<VectorXd> stateSequence; 
  std::vector<MatrixXd> covSequence;  

  std::tie(gainSequence, stateSequence, covSequence) = Kalman::kf(Z, M);
  }
