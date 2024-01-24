#include <iostream>
#include <iomanip>
#include <vector>

#include <Eigen/Dense>

#include "kalman.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

template <typename Derived>
void print_inv_cond(const MatrixBase<Derived>& a)
{
	std::cout << a << std::endl;
}

int main()
{
	int n = 3;
	int m = 1;
	int p = 1;
	int N = 20;

	MatrixXd A(n, n);
	MatrixXd B(n, p);
	MatrixXd C(m, n);
	MatrixXd D(m, p);
	MatrixXd Q(n, n);
	MatrixXd R(m, m);
	MatrixXd S(n, m);
	MatrixXd P0(n, n);
	VectorXd x0(n);
	MatrixXd U;
	MatrixXd Y;

	// Initial state
	x0 = MatrixXd::Random(n, 1);
	//std::cout << "xf: \n" << xf << std::endl;
	P0 << .1, .1, .1, .1, 10000, 10, .1, 10, 100;

	// Some random measurements
	U = MatrixXd::Random(N, p);
	Y = MatrixXd::Random(N, m);

	// Discrete LTI system
	double dt = 1.0 / 30; // Time step
	A << 1, dt, 0, 0, 1, dt, 0, 0, 1;
	B << 0, 0.1, 0.1;
	C << 1, 0, 0;
	D << 0.5;

	// Covariance matrices
	Q << .05, .05, .0, .05, .05, .0, .0, .0, .0;
	R << 5;
	S = MatrixXd::Identity(n, m);

    std::cout << "Address U:" << U.data() << std::endl;
    std::cout << "Address Y:" << Y.data() << std::endl;

	Observations Z(U, Y);
	const Observations O(U, Y);

	std::cout << "U:" << U << std::endl;
    std::cout << "Y:" << Y << std::endl;

	std::cout << "Observations.U():" << O.U() << std::endl;
    std::cout << "Observations.Y():" << O.Y() << std::endl;


    std::cout << "Address U:" << U.data() << std::endl;
	std::cout << "U:" << U << std::endl;

	LssModel M(A, B, C, D, P0, Q, R, S, x0, P0);

	const MatrixXd Ac = A;
	const MatrixXd Bc = B;
	const MatrixXd Cc = C;
	const MatrixXd Dc = D;
	const MatrixXd Qc = Q;
	const MatrixXd Rc = R;
	const MatrixXd Sc = S;
	const MatrixXd x0c = x0;
	const MatrixXd P0c = P0;
	
	// LtissModel M1(Ac, Bc, Cc, Dc, P0c, Qc, Rc, Sc, x0c, P0c);

	TestTemplate tT(A);
	TestTemplate tT2(Ac);
	std::cout << "Address A:" << A.data() << std::endl;
	// std::cout << "Address tT.A():" << tT.A().data() << std::endl;


	TestModel tM(Ac);
	std::cout << "Address A:" << A.data() << std::endl;
	std::cout << "Address tM.A():" << tM.A().data() << std::endl;
	A *= 2;
	// tM._A = &Bc; 

	std::cout << "Ac:" << Ac << std::endl;
	std::cout << "tM.A():" << tM.A() << std::endl;
	std::cout << "A:" << A << std::endl;
	// std::cout << "tT.A():" << tT.A() << std::endl;
	

	// std::cout << "Address A = Tm.A():" << A.data() == tM.A().data() << std::endl;
	
	print_inv_cond(A);

	std::vector<VectorXd> gainSequence;
	std::vector<VectorXd> stateSequence;
	std::vector<MatrixXd> covSequence;

	std::tie(gainSequence, stateSequence, covSequence) = Kalman::kf(Z, M);
	std::cout << "state: \n";
	for (int ii=0; ii<Z.U().rows(); ii++ ){
		std::cout << std::setprecision (4) << stateSequence[ii].transpose() << std::endl;
	}
}