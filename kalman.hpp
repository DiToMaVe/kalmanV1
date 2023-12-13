#include <Eigen/Dense>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#pragma once

class Measurements{

    public:
    const Eigen::MatrixXd U, Y;

    Measurements(const Eigen::MatrixXd U, const Eigen::MatrixXd Y);

};

class LtiModel{
    
    public:

    int n, m, p;
    const VectorXd x0;
    const MatrixXd A, B, C, D, P, Q, R, S, P0;
    
    LtiModel(
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

class Kalman{

    public:
        const Measurements& Z;
        const LtiModel& M;

        // Auxiliary matrices
        const MatrixXd I_n, I_m;
        MatrixXd R_inv, A_bar, B_bar, Q_bar, Q_root, R_root;

        Kalman(Measurements& Z, LtiModel& M);

        std::tuple<VectorXd, MatrixXd, MatrixXd> kalmanUpdate(VectorXd& xf, MatrixXd& Pf_root, VectorXd& u, VectorXd& y);

        void kf(Measurements &Z, LtiModel& M);

};
