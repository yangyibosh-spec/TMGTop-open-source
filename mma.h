/*
 * This file is part of TMGTop.
 *
 * The interface accompanies a C++ adaptation of the MATLAB GCMMA-MMA-code
 * developed by Krister Svanberg.
 * Copyright (C) 2006-2009 Krister Svanberg
 * Copyright (C) 2026 Yibo Yang
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once
#include <Eigen/Dense>

struct MMAResult {
    Eigen::VectorXd xmma, ymma, lam, xsi, eta, mu, s, low, upp;
    double zmma = 0.0;
    double zet = 0.0;
};

MMAResult mmasub(
    int m, int n, int iter,
    const Eigen::VectorXd& xval,
    const Eigen::VectorXd& xmin,
    const Eigen::VectorXd& xmax,
    const Eigen::VectorXd& xold1,
    const Eigen::VectorXd& xold2,
    double f0val,
    const Eigen::VectorXd& df0dx,
    const Eigen::VectorXd& fval,
    const Eigen::MatrixXd& dfdx,
    const Eigen::VectorXd& low_in,
    const Eigen::VectorXd& upp_in,
    double a0,
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& c,
    const Eigen::VectorXd& d
);

MMAResult mmasub_m1(
    int n,
    int iter,
    const Eigen::VectorXd& xval,
    const Eigen::VectorXd& xmin,
    const Eigen::VectorXd& xmax,
    const Eigen::VectorXd& xold1,
    const Eigen::VectorXd& xold2,
    double f0val,
    const Eigen::VectorXd& df0dx,
    double fval0,
    const Eigen::VectorXd& dfdx0,
    const Eigen::VectorXd& low_in,
    const Eigen::VectorXd& upp_in,
    double a0,
    double a,
    double c,
    double d
);
