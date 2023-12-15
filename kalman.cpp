#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "kalman.hpp"


Measurements::Measurements(const Eigen::MatrixXd U, const Eigen::MatrixXd Y)
	: U(U), Y(Y)
{
	// ToDo: Check on length N
}

LssModel::LssModel(
	const Eigen::MatrixXd A,
	const Eigen::MatrixXd B,
	const Eigen::MatrixXd C,
	const Eigen::MatrixXd D,
	const Eigen::MatrixXd P,
	const Eigen::MatrixXd Q,
	const Eigen::MatrixXd R,
	const Eigen::MatrixXd S,
	const Eigen::VectorXd x0,
	const Eigen::MatrixXd P0
)
	: A(A), B(B), C(C), D(D), P(P), Q(Q), R(R), S(S), x0(x0), P0(P0)
{
	n = A.rows();
	p = B.cols();
	m = C.rows();
	// ToDo: Checks on dimensions 
}

Kalman::Kalman(Measurements &Z, LssModel &M) :
	Z(Z),
	M(M),
	I_n(MatrixXd::Identity(M.n, M.n)),
	I_m(MatrixXd::Identity(M.m, M.m)),
	R_inv(M.R.ldlt().solve(I_m)), // ldlt or llt?
	A_bar(M.A - M.S*R_inv*M.C),
	B_bar(M.B - M.S*R_inv*M.D),
	Q_bar(M.Q - M.S*R_inv*M.S.transpose()),
	Q_root(M.Q.llt().matrixL()),   // robust Cholesky?
	R_root(M.R.llt().matrixL())   // robust Cholesky?
{}

std::tuple<VectorXd, VectorXd, MatrixXd> Kalman::kalmanUpdate(VectorXd &xf, MatrixXd &Pf_root, VectorXd &u, VectorXd &y)
{
	// System
	int n = M.n;
	int p = M.p;
	int m = M.m;

	// Pp
	MatrixXd QR1(2 * n, n);
	MatrixXd QR1_Q(2 * n, 2 * n);
	MatrixXd QR1_R(2 * n, n);
	QR1.block(0, 0, n, n) = Pf_root.transpose()*A_bar.transpose();
	QR1.block(n, 0, n, n) = Q_root.transpose();

	Eigen::HouseholderQR<MatrixXd> qr1(QR1);
	QR1_R = qr1.matrixQR().triangularView<Eigen::Upper>();
	QR1_Q = qr1.householderQ();
	MatrixXd Pp_root = QR1_R.block(0, 0, n, n).transpose();

	// XA=B --> A^T * X^T = B^T 
	MatrixXd K;
	MatrixXd BSR(n, m + p);
	VectorXd z(m + p);

	// BSR: the block matrix [B_bar S*R_inv]
	BSR.block(0, 0, n, p) = B_bar;
	BSR.block(0, p, n, m) = M.S*R_inv;

	z.head(p) = u;
	z.segment(p, m) = y;

	K = (M.C*Pp_root*Pp_root.transpose()*M.C.transpose() + M.R).llt().solve(M.C*Pp_root*Pp_root.transpose()).transpose();
	xf = (I_n - K * M.C)*(A_bar*xf + BSR * z) + K * (y - M.D*u);

	//Pf
	MatrixXd QR2(m + n, m + n);
	MatrixXd QR2_Q(m + n, m + n);
	MatrixXd QR2_R(m + n, m + n);

	QR2.block(0, 0, m, m) = R_root;
	QR2.block(0, m, m, n) = M.C*Pp_root;
	QR2.block(m, m, n, n) = Pp_root;

	Eigen::HouseholderQR<MatrixXd> qr2(QR2);
	QR2_R = qr2.matrixQR().triangularView<Eigen::Upper>();
	QR2_Q = qr2.householderQ();
	Pf_root = QR2_R.block(m, m, n, n).transpose();

	return std::make_tuple(K, xf, Pf_root);
}

std::tuple<std::vector<VectorXd>, std::vector<VectorXd>, std::vector<MatrixXd>> Kalman::kf(Measurements &Z, LssModel &M)
{

	Kalman kalmanObj(Z, M);
	int N = Z.Y.rows();
	std::vector<VectorXd> gainSequence;
	std::vector<VectorXd> stateSequence;
	std::vector<MatrixXd> covSequence;

	VectorXd K;
	VectorXd xf = M.x0;
	MatrixXd Pf_root = M.P0.llt().matrixL();
	for (int ii = 0; ii < N; ii++)
	{
		VectorXd u = Z.U.row(ii).transpose();
		VectorXd y = Z.Y.row(ii).transpose();
		std::tie(K, xf, Pf_root) = kalmanObj.kalmanUpdate(xf, Pf_root, u, y);

		gainSequence.push_back(K);
		stateSequence.push_back(xf);
		covSequence.push_back(Pf_root);
	}
	return std::make_tuple(gainSequence, stateSequence, covSequence);
}
