#include <Eigen/Dense>
#include <tuple>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixBase;

#pragma once

class ObservationsV1{
	/*
	Z: the data structure containing the measured outputs in Z.y and 
	possibly the measured inputs in Z.u. If the number of data
	points is N, the number of outputs is p, and the number of
	inputs is m, then Z.y is an N x p matrix and Z.u is an N x m
	matrix.*/
	// ToDo: Assess final choice N x ... or ... x N
	public:
		const Eigen::MatrixXd U, Y;
		ObservationsV1(const Eigen::MatrixXd U, const Eigen::MatrixXd Y);
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
		const MatrixXd* U_;
		const MatrixXd* Y_;
	public:
    	Observations(const MatrixXd& U, const MatrixXd& Y): U_{&U}, Y_{&Y}{};
		const MatrixXd& U() const {return *U_;};
		const MatrixXd& Y() const {return *Y_;};
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
		const VectorXd* _x0; 
		const MatrixXd* _A;
		const MatrixXd* _B;
		const MatrixXd* _C;
		const MatrixXd* _D;
		const MatrixXd* _P;
		const MatrixXd* _Q;
		const MatrixXd* _R;
		const MatrixXd* _S;
		const MatrixXd* _P0;
		
	public:
		int n, p, m;
		LtissModel(
			const MatrixXd& A,
			const MatrixXd& B,
			const MatrixXd& C,
			const MatrixXd& D,
			const MatrixXd& P,
			const MatrixXd& Q,
			const MatrixXd& R,
			const MatrixXd& S,
			const VectorXd& x0,
			const MatrixXd& P0
		);
		const VectorXd& x0() {return *_x0;};
		const MatrixXd& A() {return *_A;};
		const MatrixXd& B() {return *_B;};
		const MatrixXd& C() {return *_C;};
		const MatrixXd& D() {return *_D;};
		const MatrixXd& P() {return *_P;};
		const MatrixXd& Q() {return *_Q;};
		const MatrixXd& R() {return *_R;};
		const MatrixXd& S() {return *_S;};
		const MatrixXd& P0() {return *_P0;};
};

class Kalman {

	private:
		// Observations* _Z;
		// LtissModel* _M;

	public:
		LssModel& M;

		MatrixXd _I_n, _I_m;
		MatrixXd _R_inv, _A_bar, _B_bar, _Q_bar, _Q_root, _R_root;

		// LtissModel& M() {return *_M;};
		// Observations& Z() {return *_Z;};

		Kalman(LssModel& M);

		std::tuple<VectorXd, VectorXd, MatrixXd> update(VectorXd &xf, MatrixXd &Pf_root, VectorXd &u, VectorXd &y);

		static std::tuple<std::vector<VectorXd>, std::vector<VectorXd>, std::vector<MatrixXd>> kf(const Observations& Z, LssModel& M);
}; 

class TestModel{
	// private:
	// 	const MatrixXd* _A;
	public:
		const MatrixXd* _A;
		TestModel(const MatrixXd &A) : _A(&A) {}
		constexpr auto A() const -> const MatrixXd& {return *_A;};
};

class TestTemplate{
	// private:
	// 	const MatrixXd* _A;
	public:
		template <typename Derived>
		TestTemplate(const MatrixBase<Derived>& A){
			std::cout << "Address A:" << &A << std::endl;
		};
		// TestTemplate(MatrixXd& A): _A(&A){};
		// MatrixXd& A() {return *_A;};
};




