#include <Eigen/Dense>
#include <tuple>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#pragma once

class Measurements{
	/*
	Z: the data structure containing the measured outputs in Z.y and 
	possibly the measured inputs in Z.u. If the number of data
	points is N, the number of outputs is p, and the number of
	inputs is m, then Z.y is an N x p matrix and Z.u is an N x m
	matrix.*/
	// ToDo: Assess final choice N x ... or ... x N
	public:
		const Eigen::MatrixXd U, Y;
		Measurements(const Eigen::MatrixXd U, const Eigen::MatrixXd Y);
};

class Observations{
	/*
	Z: the data structure containing the observed outputs in Z.y and 
	possibly the observed inputs in Z.u. If the number of data
	points is N, the number of outputs is p, and the number of
	inputs is m, then Z.y is an N x p matrix and Z.u is an N x m
	matrix.*/
	// ToDo: Assess final choice N x ... or ... x N	
	private:
		MatrixXd* U_;
		MatrixXd* Y_;
	public:
    	Observations(MatrixXd& U, MatrixXd& Y);
		MatrixXd& U() {return *U_;};
		MatrixXd& Y() {return *Y_;};
};

class LssModel {
	/*	
	Linear state space (ss) model, currently only a linear time-invariant (LTI) state space model is implemented.

	For LssModel M()
	M.A,B,C,D: system matrices; for each matrix.
	M.Q,S,R: covariance matrices for process and measurement noise, respectively. 
	M.x0,P0: Initial state mean (X1) and its covariance matrix (P1), respectively.

	Dimensions:

	n: number of states
	p: number of inputs
	m: number of outputs
	
	A(n, n)
	B(n, p)
	C(n, m)
	D(m, p)
	Q(n, n)
	S(m, n)
	R(m, m)
	x1(n, 1)
	P1(n, n)

	*/

	public:

		int n, p, m;
		const VectorXd x0;
		const MatrixXd A, B, C, D, P, Q, R, S, P0;

		LssModel(
			const MatrixXd A,
			const MatrixXd B,
			const MatrixXd C,
			const MatrixXd D,
			const MatrixXd P,
			const MatrixXd Q,
			const MatrixXd R,
			const MatrixXd S,
			const VectorXd x0,
			const MatrixXd P0
		);
};

class LtissModel{
	/*	
	Linear time-invariant state space (ss) model.

	For LtissModel M()
	M.A,B,C,D: system matrices; for each matrix.
	M.Q,S,R: covariance matrices for process and measurement noise, respectively. 
	M.x0,P0: Initial state mean (X1) and its covariance matrix (P1), respectively.

	Dimensions:

	n: number of states
	p: number of inputs
	m: number of outputs
	
	A(n, n)
	B(n, p)
	C(n, m)
	D(m, p)
	Q(n, n)
	S(m, n)
	R(m, m)
	x1(n, 1)
	P1(n, n)
	*/
	private:
		VectorXd* x0_; 
		MatrixXd* A_;
		MatrixXd* B_;
		MatrixXd* C_;
		MatrixXd* D_;
		MatrixXd* P_;
		MatrixXd* Q_;
		MatrixXd* R_;
		MatrixXd* S_;
		MatrixXd* P0_;
	public:
		int n, p, m;
		LtissModel(
			MatrixXd& A,
			MatrixXd& B,
			MatrixXd& C,
			MatrixXd& D,
			MatrixXd& P,
			MatrixXd& Q,
			MatrixXd& R,
			MatrixXd& S,
			VectorXd& x0,
			MatrixXd& P0
		);
		VectorXd& x0() {return *x0_;};
		MatrixXd& A() {return *A_;};
		MatrixXd& B() {return *B_;};
		MatrixXd& C() {return *C_;};
		MatrixXd& D() {return *D_;};
		MatrixXd& P() {return *P_;};
		MatrixXd& Q() {return *Q_;};
		MatrixXd& R() {return *R_;};
		MatrixXd& S() {return *S_;};
		MatrixXd& P0() {return *P0_;};
};

class Kalman {

	public:
		const Measurements& Z;
		const LssModel& M;

		const MatrixXd I_n, I_m;
		const MatrixXd R_inv, A_bar, B_bar, Q_bar, Q_root, R_root;

		Kalman(Measurements &Z, LssModel &M);

		std::tuple<VectorXd, VectorXd, MatrixXd> kalmanUpdate(VectorXd &xf, MatrixXd &Pf_root, VectorXd &u, VectorXd &y);

		static std::tuple<std::vector<VectorXd>, std::vector<VectorXd>, std::vector<MatrixXd>> kf(Measurements &Z, LssModel &M);
}; 
