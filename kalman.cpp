#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "kalman.hpp"

using Eigen::MatrixBase;


ObservationsV1::ObservationsV1(const Eigen::MatrixXd U, const Eigen::MatrixXd Y): U(U), Y(Y)
{
	// ToDo: Check on length N
	std::cout << "Inside class Measurements." << std::endl;
	std::cout << "Address U:" << U.data() << std::endl;
    std::cout << "Address Y:" << Y.data() << std::endl;

}

LssModel::LssModel(
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
): A(A), B(B), C(C), D(D), P(P), Q(Q), R(R), S(S), x0(x0), P0(P0)
{
	n = A.rows();
	p = B.cols();
	m = C.rows();
	// ToDo: Checks on dimensions 
}

LtissModel::LtissModel(
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
)
: _A(&A), _B(&B), _C(&C), _D(&D), _P(&P), _Q(&Q), _R(&R), _S(&S), _x0(&x0), _P0(&P0)
{
	n = A.rows();
	p = B.cols();
	m = C.rows();
	// ToDo: Checks on dimensions 
}

Kalman::Kalman(LssModel& M) 
:	M(M),
	_I_n(MatrixXd::Identity(M.n, M.n)),
	_I_m(MatrixXd::Identity(M.m, M.m)),
	_R_inv(M.R.ldlt().solve(_I_m)), // ldlt or llt?
	_A_bar(M.A - M.S*_R_inv*M.C),
	_B_bar(M.B - M.S*_R_inv*M.D),
	_Q_bar(M.Q - M.S*_R_inv*M.S.transpose()),
	_Q_root(M.Q.llt().matrixL()),   // robust Cholesky?
	_R_root(M.R.llt().matrixL())   // robust Cholesky?
{}

std::tuple<VectorXd, VectorXd, MatrixXd> Kalman::update(VectorXd &xf, MatrixXd &Pf_root, VectorXd &u, VectorXd &y)
{
	// System
	int n = M.n;
	int p = M.p;
	int m = M.m;

	// Pp
	MatrixXd QR1(2 * n, n);
	MatrixXd QR1_Q(2 * n, 2 * n);
	MatrixXd QR1_R(2 * n, n);
	QR1.block(0, 0, n, n) = Pf_root.transpose()*_A_bar.transpose();
	QR1.block(n, 0, n, n) = _Q_root.transpose();

	Eigen::HouseholderQR<MatrixXd> qr1(QR1);
	QR1_R = qr1.matrixQR().triangularView<Eigen::Upper>();
	QR1_Q = qr1.householderQ();
	MatrixXd Pp_root = QR1_R.block(0, 0, n, n).transpose();

	// XA=B --> A^T * X^T = B^T 
	MatrixXd K;
	MatrixXd BSR(n, m + p);
	VectorXd z(m + p);

	// BSR: the block matrix [B_bar S*R_inv]
	BSR.block(0, 0, n, p) = _B_bar;
	BSR.block(0, p, n, m) = M.S*_R_inv;

	z.head(p) = u;
	z.segment(p, m) = y;

	K = (M.C*Pp_root*Pp_root.transpose()*M.C.transpose() + M.R).llt().solve(M.C*Pp_root*Pp_root.transpose()).transpose();
	xf = (_I_n - K * M.C)*(_A_bar*xf + BSR * z) + K * (y - M.D*u);

	//Pf
	MatrixXd QR2(m + n, m + n);
	MatrixXd QR2_Q(m + n, m + n);
	MatrixXd QR2_R(m + n, m + n);

	QR2.block(0, 0, m, m) = _R_root;
	QR2.block(0, m, m, n) = M.C*Pp_root;
	QR2.block(m, m, n, n) = Pp_root;

	Eigen::HouseholderQR<MatrixXd> qr2(QR2);
	QR2_R = qr2.matrixQR().triangularView<Eigen::Upper>();
	QR2_Q = qr2.householderQ();
	Pf_root = QR2_R.block(m, m, n, n).transpose();

	return std::make_tuple(K, xf, Pf_root);
}

std::tuple<std::vector<VectorXd>, std::vector<VectorXd>, std::vector<MatrixXd>> Kalman::kf(const Observations& Z, LssModel& M)
{
	Kalman kalmanObj(M);
	int N = Z.Y().rows();
	std::vector<VectorXd> gainSequence;
	std::vector<VectorXd> stateSequence;
	std::vector<MatrixXd> covSequence;

	VectorXd K;
	VectorXd xf = M.x0;
	MatrixXd Pf_root = M.P0.llt().matrixL();
	for (int ii = 0; ii < N; ii++)
	{
		VectorXd u = Z.U().row(ii).transpose();
		VectorXd y = Z.Y().row(ii).transpose();
		std::tie(K, xf, Pf_root) = kalmanObj.update(xf, Pf_root, u, y);

		gainSequence.push_back(K);
		stateSequence.push_back(xf);
		covSequence.push_back(Pf_root);
	}
	return std::make_tuple(gainSequence, stateSequence, covSequence);
}

