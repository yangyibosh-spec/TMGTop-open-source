#include "mma.h"
/*
 * This file is part of TMGTop.
 *
 * Portions of this implementation are based on the MATLAB GCMMA-MMA-code
 * developed by Krister Svanberg and distributed under the GNU General
 * Public License, version 3 or (at your option) any later version.
 * Copyright (C) 2006-2009 Krister Svanberg
 *
 * Translated to C++ and subsequently modified for TMGTop in 2026.
 * C++ adaptation and modifications:
 * Copyright (C) 2026 Yibo Yang
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * TMGTop is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * TMGTop is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with TMGTop. If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <limits>
#include <chrono>
#include <cstdlib>
#include <sstream>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using std::array;
using std::cout;
using std::endl;
using std::size_t;
using std::string;
using std::vector;

static inline void accum_stats_vec(const Eigen::VectorXd& v, double& sqsum, double& maxabs)
{
    sqsum += v.squaredNorm();
    double m = v.cwiseAbs().maxCoeff();
    if (m > maxabs) maxabs = m;
}

static inline void accum_stats_scalar(double v, double& sqsum, double& maxabs)
{
    sqsum += v * v;
    double a = std::abs(v);
    if (a > maxabs) maxabs = a;
}

static double mma_env_nonnegative_double(const char* name, double fallback)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    try {
        const double value = std::stod(raw);
        return (std::isfinite(value) && value >= 0.0) ? value : fallback;
    }
    catch (...) {
        return fallback;
    }
}

static int mma_env_positive_int(const char* name, int fallback)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    try {
        const int value = std::stoi(raw);
        return value > 0 ? value : fallback;
    }
    catch (...) {
        return fallback;
    }
}

static bool mma_use_scalar_m1_solver()
{
    const char* raw = std::getenv("TMGTOP_MMA_SOLVER");
    if (raw == nullptr || *raw == '\0') return true;
    const std::string mode(raw);
    return !(mode == "primal_dual" || mode == "pd" || mode == "legacy");
}

// Exact separable solution of the m=1 MMA subproblem used by TMGTop.
// The production configuration has a=0 and d=0, hence the only coupled
// unknown is the scalar multiplier lambda of the volume constraint.  For a
// fixed lambda every x_j has a closed-form minimizer.  A safeguarded bisection
// therefore replaces the O(n)-unknown primal-dual Newton system.
static MMAResult subsolv_m1_scalar_dual(
    int n,
    const Eigen::VectorXd& low,
    const Eigen::VectorXd& upp,
    const Eigen::VectorXd& alfa,
    const Eigen::VectorXd& beta,
    const Eigen::VectorXd& p0,
    const Eigen::VectorXd& q0,
    const Eigen::VectorXd& P,
    const Eigen::VectorXd& Q,
    double a0,
    double a,
    double b,
    double c,
    double d)
{
    using Eigen::Index;
    using Eigen::VectorXd;

    if (n <= 0 || low.size() != n || upp.size() != n || alfa.size() != n ||
        beta.size() != n || p0.size() != n || q0.size() != n ||
        P.size() != n || Q.size() != n) {
        throw std::invalid_argument("MMA scalar-dual solver received inconsistent vector sizes.");
    }
    if (std::abs(a) > 1e-14 || std::abs(d) > 1e-14 || !(a0 > 0.0) || !(c > 0.0)) {
        throw std::invalid_argument(
            "MMA scalar-dual solver requires the TMGTop m=1 setting a=0, d=0, a0>0, c>0.");
    }

    const auto t0 = std::chrono::steady_clock::now();
    const int max_bisection = mma_env_positive_int("TMGTOP_MMA_DUAL_MAX_IT", 60);
    const double relative_constraint_tol =
        mma_env_nonnegative_double("TMGTOP_MMA_DUAL_RTOL", 1e-10);
    const double constraint_tol =
        std::max(1e-8, relative_constraint_tol * std::max(1.0, std::abs(b)));
    const double tiny = 1e-300;

    VectorXd x(n);

    auto evaluate = [&](double lambda, bool store_x) -> double {
        double gsum = 0.0;
        const double* lowp = low.data();
        const double* uppp = upp.data();
        const double* alfap = alfa.data();
        const double* betap = beta.data();
        const double* p0p = p0.data();
        const double* q0p = q0.data();
        const double* pp = P.data();
        const double* qp = Q.data();
        double* xp = x.data();

#if defined(_OPENMP)
#pragma omp parallel for reduction(+:gsum) schedule(static)
#endif
        for (Index i = 0; i < static_cast<Index>(n); ++i) {
            const double plam = std::max(tiny, p0p[i] + lambda * pp[i]);
            const double qlam = std::max(tiny, q0p[i] + lambda * qp[i]);
            const double sqrtp = std::sqrt(plam);
            const double sqrtq = std::sqrt(qlam);
            double xi = (sqrtq * uppp[i] + sqrtp * lowp[i]) / (sqrtp + sqrtq);
            xi = std::max(alfap[i], std::min(betap[i], xi));
            const double ux = std::max(tiny, uppp[i] - xi);
            const double xl = std::max(tiny, xi - lowp[i]);
            gsum += pp[i] / ux + qp[i] / xl;
            if (store_x) xp[i] = xi;
        }
        return gsum - b;
    };

    int dual_iterations = 0;
    double lambda = 0.0;
    double y = 0.0;
    double residual = evaluate(0.0, false);

    if (residual > constraint_tol) {
        double lambda_lo = 0.0;
        double lambda_hi = std::min(1.0, c);
        double residual_hi = evaluate(lambda_hi, false);

        while (residual_hi > 0.0 && lambda_hi < c) {
            lambda_lo = lambda_hi;
            lambda_hi = std::min(c, 2.0 * lambda_hi);
            residual_hi = evaluate(lambda_hi, false);
        }

        if (residual_hi > 0.0 && lambda_hi >= c) {
            // The MMA slack y becomes active at lambda=c.  This is a valid
            // subproblem solution, but is reported because it means that the
            // current move limits cannot satisfy the approximated constraint.
            lambda = c;
            residual = residual_hi;
            y = residual_hi;
        }
        else {
            for (dual_iterations = 0; dual_iterations < max_bisection; ++dual_iterations) {
                lambda = 0.5 * (lambda_lo + lambda_hi);
                residual = evaluate(lambda, false);
                if (std::abs(residual) <= constraint_tol ||
                    (lambda_hi - lambda_lo) <= 1e-12 * std::max(1.0, lambda_hi)) {
                    break;
                }
                if (residual > 0.0) lambda_lo = lambda;
                else lambda_hi = lambda;
            }
        }
    }

    const double raw_constraint = evaluate(lambda, true);
    if (lambda < c || y == 0.0) y = 0.0;
    if (lambda >= c && raw_constraint > 0.0) y = raw_constraint;
    const double signed_constraint = raw_constraint - y;
    const double primal_violation = std::max(0.0, signed_constraint);

    VectorXd xsi = VectorXd::Zero(n);
    VectorXd eta = VectorXd::Zero(n);
    double stationarity_max = 0.0;

    const double* lowp = low.data();
    const double* uppp = upp.data();
    const double* alfap = alfa.data();
    const double* betap = beta.data();
    const double* p0p = p0.data();
    const double* q0p = q0.data();
    const double* pp = P.data();
    const double* qp = Q.data();
    const double* xp = x.data();
    double* xsip = xsi.data();
    double* etap = eta.data();

#if defined(_OPENMP)
#pragma omp parallel for reduction(max:stationarity_max) schedule(static)
#endif
    for (Index i = 0; i < static_cast<Index>(n); ++i) {
        const double ux = std::max(tiny, uppp[i] - xp[i]);
        const double xl = std::max(tiny, xp[i] - lowp[i]);
        const double plam = p0p[i] + lambda * pp[i];
        const double qlam = q0p[i] + lambda * qp[i];
        const double derivative = plam / (ux * ux) - qlam / (xl * xl);
        const double bound_tol = 1e-11 * std::max(1.0, betap[i] - alfap[i]);
        double violation = 0.0;
        if (xp[i] <= alfap[i] + bound_tol) {
            xsip[i] = std::max(0.0, derivative);
            violation = std::max(0.0, -derivative);
        }
        else if (xp[i] >= betap[i] - bound_tol) {
            etap[i] = std::max(0.0, -derivative);
            violation = std::max(0.0, derivative);
        }
        else {
            violation = std::abs(derivative);
        }
        stationarity_max = std::max(stationarity_max, violation);
    }

    if (!x.allFinite() || !std::isfinite(lambda) || !std::isfinite(y) ||
        primal_violation > 10.0 * constraint_tol) {
        std::ostringstream oss;
        oss << "MMA scalar-dual solve failed: finite=" << x.allFinite()
            << ", lambda=" << lambda
            << ", primalViolation=" << primal_violation
            << ", tolerance=" << constraint_tol;
        throw std::runtime_error(oss.str());
    }

    MMAResult out;
    out.xmma = x;
    out.ymma = VectorXd::Constant(1, y);
    out.zmma = 0.0;
    out.lam = VectorXd::Constant(1, lambda);
    out.xsi = xsi;
    out.eta = eta;
    out.mu = VectorXd::Constant(1, std::max(0.0, c - lambda));
    out.zet = a0;
    out.s = VectorXd::Constant(1, std::max(0.0, -signed_constraint));

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "MMA_DIAG solver=scalar_dual_m1"
              << " n=" << n
              << " dualIts=" << dual_iterations
              << " lambda=" << lambda
              << " constraintResidual=" << signed_constraint
              << " constraintTol=" << constraint_tol
              << " stationarityMax=" << stationarity_max
              << " slackY=" << y
              << " elapsed=" << elapsed
              << std::endl;
    return out;
}


MMAResult subsolv_m1(
    int n,
    double epsimin,
    const Eigen::VectorXd& low,
    const Eigen::VectorXd& upp,
    const Eigen::VectorXd& alfa,
    const Eigen::VectorXd& beta,
    const Eigen::VectorXd& p0,
    const Eigen::VectorXd& q0,
    const Eigen::VectorXd& P,   // size n, not MatrixXd anymore
    const Eigen::VectorXd& Q,   // size n
    double a0,
    double a,
    double b,
    double c,
    double d)
{
    using Eigen::VectorXd;

    const VectorXd een = VectorXd::Ones(n);

    double epsi = 1.0;

    VectorXd x = 0.5 * (alfa + beta);
    double y = 1.0;
    double z = 1.0;
    double lam = 1.0;
    VectorXd xsi = (x - alfa).cwiseInverse().cwiseMax(een);
    VectorXd eta = (beta - x).cwiseInverse().cwiseMax(een);
    double mu = std::max(0.5 * c, 1.0);
    double zet = 1.0;
    double s = 1.0;

    // Ԥ乤
    VectorXd epsvecn(n);
    VectorXd ux1(n), xl1(n), ux2(n), xl2(n), ux3(n), xl3(n);
    VectorXd uxinv1(n), xlinv1(n), uxinv2(n), xlinv2(n);
    VectorXd plam(n), qlam(n), dpsidx(n), GG(n);
    VectorXd rex(n), rexsi(n), reeta(n);
    VectorXd delx(n), diagx(n), diagxinv(n);
    VectorXd dx(n), dxsi(n), deta(n);

    VectorXd xold(n), xsiold(n), etaold(n);

    double rey = 0.0, rez = 0.0, relam = 0.0, remu = 0.0, rezet = 0.0, res = 0.0;

    const auto mma_t0 = std::chrono::steady_clock::now();
    // Zero disables the wall-clock limit.  If a positive limit is requested,
    // timeout is a hard error: returning a partially solved MMA step is never
    // allowed.  Example: TMGTOP_MMA_TIME_LIMIT_SEC=600.
    const double mma_time_limit_sec =
        mma_env_nonnegative_double("TMGTOP_MMA_TIME_LIMIT_SEC", 0.0);
    const int max_newton_per_barrier =
        mma_env_positive_int("TMGTOP_MMA_PD_MAX_NEWTON", 80);
    int total_newton_iterations = 0;
    double final_residumax = std::numeric_limits<double>::infinity();
    double final_barrier = epsi;

    auto compute_residual_stats = [&](double epsi_local, double& residunorm, double& residumax)
        {
            epsvecn.setConstant(epsi_local);

            ux1 = upp - x;
            xl1 = x - low;
            ux2 = ux1.array().square().matrix();
            xl2 = xl1.array().square().matrix();
            uxinv1 = ux1.cwiseInverse();
            xlinv1 = xl1.cwiseInverse();

            plam.noalias() = p0 + lam * P;
            qlam.noalias() = q0 + lam * Q;

            double gvec = P.dot(uxinv1) + Q.dot(xlinv1);
            dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);

            rex = dpsidx - xsi + eta;
            rey = c + d * y - mu - lam;
            rez = a0 - zet - a * lam;
            relam = gvec - a * z - y + s - b;
            rexsi = xsi.cwiseProduct(x - alfa) - epsvecn;
            reeta = eta.cwiseProduct(beta - x) - epsvecn;
            remu = mu * y - epsi_local;
            rezet = zet * z - epsi_local;
            res = lam * s - epsi_local;

            double sqsum = 0.0;
            double maxabs = 0.0;

            accum_stats_vec(rex, sqsum, maxabs);
            accum_stats_scalar(rey, sqsum, maxabs);
            accum_stats_scalar(rez, sqsum, maxabs);
            accum_stats_scalar(relam, sqsum, maxabs);
            accum_stats_vec(rexsi, sqsum, maxabs);
            accum_stats_vec(reeta, sqsum, maxabs);
            accum_stats_scalar(remu, sqsum, maxabs);
            accum_stats_scalar(rezet, sqsum, maxabs);
            accum_stats_scalar(res, sqsum, maxabs);

            residunorm = std::sqrt(sqsum);
            residumax = maxabs;
        };

    while (epsi > epsimin) {
        double residunorm = 0.0, residumax = 0.0;
        compute_residual_stats(epsi, residunorm, residumax);

        int ittt = 0;
        while (residumax > 0.9 * epsi && ittt < max_newton_per_barrier) {
            ++ittt;
            ++total_newton_iterations;
            double mma_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - mma_t0).count();
            if (mma_time_limit_sec > 0.0 && mma_elapsed > mma_time_limit_sec) {
                std::ostringstream oss;
                oss << "MMA primal-dual solve timed out after " << mma_elapsed
                    << " s at barrier=" << epsi
                    << ", residualMax=" << residumax
                    << ". A partial MMA update was not accepted.";
                throw std::runtime_error(oss.str());
            }

            epsvecn.setConstant(epsi);

            ux1 = upp - x;
            xl1 = x - low;
            ux2 = ux1.array().square().matrix();
            xl2 = xl1.array().square().matrix();
            ux3 = (ux1.array() * ux2.array()).matrix();
            xl3 = (xl1.array() * xl2.array()).matrix();

            uxinv1 = ux1.cwiseInverse();
            xlinv1 = xl1.cwiseInverse();
            uxinv2 = ux2.cwiseInverse();
            xlinv2 = xl2.cwiseInverse();

            plam.noalias() = p0 + lam * P;
            qlam.noalias() = q0 + lam * Q;

            double gvec = P.dot(uxinv1) + Q.dot(xlinv1);
            GG = P.cwiseProduct(uxinv2) - Q.cwiseProduct(xlinv2);

            dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);

            delx = dpsidx
                - epsvecn.cwiseQuotient(x - alfa)
                + epsvecn.cwiseQuotient(beta - x);

            double dely = c + d * y - lam - epsi / y;
            double delz = a0 - a * lam - epsi / z;
            double dellam = gvec - a * z - y - b + epsi / lam;

            diagx =
                2.0 * (plam.cwiseQuotient(ux3) + qlam.cwiseQuotient(xl3)) +
                xsi.cwiseQuotient(x - alfa) +
                eta.cwiseQuotient(beta - x);

            diagxinv = diagx.cwiseInverse();

            double diagy = d + mu / y;
            double diagyinv = 1.0 / diagy;
            double diaglam = s / lam;
            double diaglamyi = diaglam + diagyinv;

            double blam = dellam + dely / diagy - GG.dot(delx.cwiseQuotient(diagx));
            double Alam = diaglamyi + (GG.array().square() * diagxinv.array()).sum();

            // 2x2 ϵͳֱ⣬ fullPivLu
            // [ Alam   a      ] [dlam] = [blam]
            // [ a    -zet/z   ] [ dz ]   [delz]
            double aa00 = Alam;
            double aa01 = a;
            double aa11 = -zet / z;

            double det = aa00 * aa11 - aa01 * aa01;
            if (std::abs(det) < 1e-30) {
                det = (det >= 0.0 ? 1e-30 : -1e-30);
            }

            double dlam = (blam * aa11 - aa01 * delz) / det;
            double dz = (aa00 * delz - aa01 * blam) / det;

            dx = -(delx + GG * dlam).cwiseQuotient(diagx);
            double dy = (dlam - dely) / diagy;

            dxsi = -xsi
                + epsvecn.cwiseQuotient(x - alfa)
                - xsi.cwiseProduct(dx).cwiseQuotient(x - alfa);

            deta = -eta
                + epsvecn.cwiseQuotient(beta - x)
                + eta.cwiseProduct(dx).cwiseQuotient(beta - x);

            double dmu = -mu + epsi / y - mu * dy / y;
            double dzet = -zet + epsi / z - zet * dz / z;
            double ds = -s + epsi / lam - s * dlam / lam;

            // ƣٹ xx / dxx
            double stmxx = 0.0;
            stmxx = std::max(stmxx, -1.01 * dy / y);
            stmxx = std::max(stmxx, -1.01 * dz / z);
            stmxx = std::max(stmxx, -1.01 * dlam / lam);
            stmxx = std::max(stmxx, (-1.01 * dxsi.array() / xsi.array()).maxCoeff());
            stmxx = std::max(stmxx, (-1.01 * deta.array() / eta.array()).maxCoeff());
            stmxx = std::max(stmxx, -1.01 * dmu / mu);
            stmxx = std::max(stmxx, -1.01 * dzet / zet);
            stmxx = std::max(stmxx, -1.01 * ds / s);

            double stmalfa = (-1.01 * dx.array() / (x - alfa).array()).maxCoeff();
            double stmbeta = (1.01 * dx.array() / (beta - x).array()).maxCoeff();

            double stminv = std::max(1.0, std::max(stmxx, std::max(stmalfa, stmbeta)));
            double steg = 1.0 / stminv;

            // ֵ
            xold = x;
            double yold = y;
            double zold = z;
            double lamold = lam;
            xsiold = xsi;
            etaold = eta;
            double muold = mu;
            double zetold = zet;
            double sold = s;

            double resinew = 2.0 * residunorm;
            int itto = 0;

            while (resinew > residunorm && itto < 8) {
                ++itto;

                x = xold + steg * dx;
                y = yold + steg * dy;
                z = zold + steg * dz;
                lam = lamold + steg * dlam;
                xsi = xsiold + steg * dxsi;
                eta = etaold + steg * deta;
                mu = muold + steg * dmu;
                zet = zetold + steg * dzet;
                s = sold + steg * ds;

                double new_norm = 0.0, new_max = 0.0;
                compute_residual_stats(epsi, new_norm, new_max);

                resinew = new_norm;
                if (resinew > residunorm) {
                    steg *= 0.5;
                }
                else {
                    residunorm = new_norm;
                    residumax = new_max;
                }
                if (dx.cwiseAbs().maxCoeff() < 1e-6 && std::abs(dlam) < 1e-8) break;
            }
        }

        final_residumax = residumax;
        final_barrier = epsi;
        if (!(residumax <= 0.9 * epsi) || !std::isfinite(residumax)) {
            std::ostringstream oss;
            oss << "MMA primal-dual solve did not converge at barrier=" << epsi
                << " after " << ittt << " Newton iterations"
                << ", residualMax=" << residumax
                << ". A partial MMA update was not accepted.";
            throw std::runtime_error(oss.str());
        }

        epsi *= 0.1;
    }

    MMAResult out;
    out.xmma = x;

    out.ymma = Eigen::VectorXd::Constant(1, y);
    out.zmma = z;
    out.lam = Eigen::VectorXd::Constant(1, lam);
    out.xsi = xsi;
    out.eta = eta;
    out.mu = Eigen::VectorXd::Constant(1, mu);
    out.zet = zet;
    out.s = Eigen::VectorXd::Constant(1, s);

    const double mma_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - mma_t0).count();
    std::cout << "MMA_DIAG solver=primal_dual_m1"
              << " n=" << n
              << " newtonIts=" << total_newton_iterations
              << " finalBarrier=" << final_barrier
              << " residualMax=" << final_residumax
              << " elapsed=" << mma_elapsed
              << std::endl;
    return out;
}

MMAResult mmasub_m1(
        int n,
        int iter,
        const Eigen::VectorXd & xval,
        const Eigen::VectorXd & xmin,
        const Eigen::VectorXd & xmax,
        const Eigen::VectorXd & xold1,
        const Eigen::VectorXd & xold2,
        double f0val,
        const Eigen::VectorXd & df0dx,
        double fval0,
        const Eigen::VectorXd & dfdx0,
        const Eigen::VectorXd & low_in,
        const Eigen::VectorXd & upp_in,
        double a0,
        double a,
        double c,
        double d)
    {
        (void)f0val;

        using Eigen::VectorXd;

        const double epsimin = 1e-5;
        const double raa0 = 1e-5;
        const double move = 0.2;
        const double albefa = 0.1;
        const double asyinit = 0.1;
        const double asyincr = 1.2;
        const double asydecr = 0.7;

        VectorXd eeen = VectorXd::Ones(n);

        VectorXd low = low_in;
        VectorXd upp = upp_in;

        if (iter < 3) {
            low = xval - asyinit * (xmax - xmin);
            upp = xval + asyinit * (xmax - xmin);
        }
        else {
            VectorXd zzz = (xval - xold1).cwiseProduct(xold1 - xold2);
            VectorXd factor = VectorXd::Ones(n);
            for (int i = 0; i < n; ++i) {
                if (zzz(i) > 0.0) factor(i) = asyincr;
                else if (zzz(i) < 0.0) factor(i) = asydecr;
            }

            low = xval - factor.cwiseProduct(xold1 - low);
            upp = xval + factor.cwiseProduct(upp - xold1);

            VectorXd lowmin = xval - 10.0 * (xmax - xmin);
            VectorXd lowmax = xval - 0.01 * (xmax - xmin);
            VectorXd uppmin = xval + 0.01 * (xmax - xmin);
            VectorXd uppmax = xval + 10.0 * (xmax - xmin);

            low = low.cwiseMax(lowmin).cwiseMin(lowmax);
            upp = upp.cwiseMin(uppmax).cwiseMax(uppmin);
        }

        VectorXd alfa = (low + albefa * (xval - low))
            .cwiseMax(xval - move * (xmax - xmin))
            .cwiseMax(xmin);

        VectorXd beta = (upp - albefa * (upp - xval))
            .cwiseMin(xval + move * (xmax - xmin))
            .cwiseMin(xmax);

        VectorXd xmami = (xmax - xmin).cwiseMax(1e-5 * eeen);
        VectorXd xmamiinv = xmami.cwiseInverse();

        VectorXd ux1 = upp - xval;
        VectorXd ux2 = ux1.array().square().matrix();
        VectorXd xl1 = xval - low;
        VectorXd xl2 = xl1.array().square().matrix();

        VectorXd uxinv = ux1.cwiseInverse();
        VectorXd xlinv = xl1.cwiseInverse();

        VectorXd p0 = df0dx.cwiseMax(0.0);
        VectorXd q0 = (-df0dx).cwiseMax(0.0);

        VectorXd pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv;
        p0 = (p0 + pq0).cwiseProduct(ux2);
        q0 = (q0 + pq0).cwiseProduct(xl2);

        // m=1 ʱP/Q ֱӾ
        VectorXd P = dfdx0.cwiseMax(0.0);
        VectorXd Q = (-dfdx0).cwiseMax(0.0);

        VectorXd PQ = 0.001 * (P + Q) + raa0 * xmamiinv;
        P += PQ;
        Q += PQ;

        P = P.cwiseProduct(ux2);
        Q = Q.cwiseProduct(xl2);

        double b = P.dot(uxinv) + Q.dot(xlinv) - fval0;

        MMAResult out;
        if (mma_use_scalar_m1_solver()) {
            out = subsolv_m1_scalar_dual(
                n,
                low, upp, alfa, beta,
                p0, q0, P, Q,
                a0, a, b, c, d
            );
        }
        else {
            out = subsolv_m1(
                n, epsimin,
                low, upp, alfa, beta,
                p0, q0, P, Q,
                a0, a, b, c, d
            );
        }

        out.low = low;
        out.upp = upp;
        return out;
    }


    static MMAResult subsolv_generic(int m, int n, double epsimin,
        const Eigen::VectorXd & low, const Eigen::VectorXd & upp,
        const Eigen::VectorXd & alfa, const Eigen::VectorXd & beta,
        const Eigen::VectorXd & p0, const Eigen::VectorXd & q0,
        const Eigen::MatrixXd & P, const Eigen::MatrixXd & Q,
        double a0, const Eigen::VectorXd & a, const Eigen::VectorXd & b,
        const Eigen::VectorXd & c, const Eigen::VectorXd & d)
    {
        if (!(m < n)) {
            throw std::runtime_error("This C++ port currently supports the m < n branch of subsolv, which matches your case m=1.");
        }

        using Eigen::VectorXd;
        using Eigen::MatrixXd;

        const VectorXd een = VectorXd::Ones(n);
        const VectorXd eem = VectorXd::Ones(m);

        const int nres = 3 * n + 4 * m + 2;   // residu total size
        const int nxx = 2 * n + 4 * m + 2;   // [y,z,lam,xsi,eta,mu,zet,s]

        double epsi = 1.0;

        VectorXd x = 0.5 * (alfa + beta);
        VectorXd y = eem;
        double   z = 1.0;
        VectorXd lam = eem;
        VectorXd xsi = (x - alfa).cwiseInverse().cwiseMax(een);
        VectorXd eta = (beta - x).cwiseInverse().cwiseMax(een);
        VectorXd mu = (0.5 * c).cwiseMax(eem);
        double   zet = 1.0;
        VectorXd s = eem;

        while (epsi > epsimin) {
            VectorXd epsvecn = epsi * een;
            VectorXd epsvecm = epsi * eem;

            VectorXd ux1 = upp - x;
            VectorXd xl1 = x - low;
            VectorXd ux2 = ux1.array().square().matrix();
            VectorXd xl2 = xl1.array().square().matrix();
            VectorXd uxinv1 = ux1.cwiseInverse();
            VectorXd xlinv1 = xl1.cwiseInverse();

            VectorXd plam = p0 + P.transpose() * lam;
            VectorXd qlam = q0 + Q.transpose() * lam;
            VectorXd gvec = P * uxinv1 + Q * xlinv1;
            VectorXd dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);

            VectorXd rex = dpsidx - xsi + eta;
            VectorXd rey = c + d.cwiseProduct(y) - mu - lam;
            double   rez = a0 - zet - a.dot(lam);
            VectorXd relam = gvec - a * z - y + s - b;
            VectorXd rexsi = xsi.cwiseProduct(x - alfa) - epsvecn;
            VectorXd reeta = eta.cwiseProduct(beta - x) - epsvecn;
            VectorXd remu = mu.cwiseProduct(y) - epsvecm;
            double   rezet = zet * z - epsi;
            VectorXd res = lam.cwiseProduct(s) - epsvecm;

            VectorXd residu(nres);
            int off = 0;
            residu.segment(off, n) = rex;    off += n;
            residu.segment(off, m) = rey;    off += m;
            residu(off++) = rez;
            residu.segment(off, m) = relam;  off += m;
            residu.segment(off, n) = rexsi;  off += n;
            residu.segment(off, n) = reeta;  off += n;
            residu.segment(off, m) = remu;   off += m;
            residu(off++) = rezet;
            residu.segment(off, m) = res;    off += m;

            if (off != nres) {
                throw std::runtime_error("subsolv: residu assembly size mismatch.");
            }

            double residunorm = residu.norm();
            double residumax = residu.cwiseAbs().maxCoeff();

            int ittt = 0;
            while (residumax > 0.9 * epsi && ittt < 40) {
                ++ittt;

                ux1 = upp - x;
                xl1 = x - low;
                ux2 = ux1.array().square().matrix();
                xl2 = xl1.array().square().matrix();
                VectorXd ux3 = (ux1.array() * ux2.array()).matrix();
                VectorXd xl3 = (xl1.array() * xl2.array()).matrix();

                uxinv1 = ux1.cwiseInverse();
                xlinv1 = xl1.cwiseInverse();
                VectorXd uxinv2 = ux2.cwiseInverse();
                VectorXd xlinv2 = xl2.cwiseInverse();

                plam = p0 + P.transpose() * lam;
                qlam = q0 + Q.transpose() * lam;
                gvec = P * uxinv1 + Q * xlinv1;

                MatrixXd GG = P * uxinv2.asDiagonal() - Q * xlinv2.asDiagonal();

                dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);

                VectorXd delx = dpsidx - epsvecn.cwiseQuotient(x - alfa) + epsvecn.cwiseQuotient(beta - x);
                VectorXd dely = c + d.cwiseProduct(y) - lam - epsvecm.cwiseQuotient(y);
                double   delz = a0 - a.dot(lam) - epsi / z;
                VectorXd dellam = gvec - a * z - y - b + epsvecm.cwiseQuotient(lam);

                VectorXd diagx =
                    2.0 * (plam.cwiseQuotient(ux3) + qlam.cwiseQuotient(xl3)) +
                    xsi.cwiseQuotient(x - alfa) + eta.cwiseQuotient(beta - x);

                VectorXd diagxinv = diagx.cwiseInverse();
                VectorXd diagy = d + mu.cwiseQuotient(y);
                VectorXd diagyinv = diagy.cwiseInverse();
                VectorXd diaglam = s.cwiseQuotient(lam);
                VectorXd diaglamyi = diaglam + diagyinv;

                VectorXd blam = dellam + dely.cwiseQuotient(diagy) - GG * (delx.cwiseQuotient(diagx));

                VectorXd bb(m + 1);
                bb.head(m) = blam;
                bb(m) = delz;

                Eigen::MatrixXd Alam = Eigen::MatrixXd(diaglamyi.asDiagonal());
                Alam.noalias() += GG * diagxinv.asDiagonal() * GG.transpose();

                MatrixXd AA(m + 1, m + 1);
                AA.setZero();
                AA.topLeftCorner(m, m) = Alam;
                AA.topRightCorner(m, 1) = a;
                AA.bottomLeftCorner(1, m) = a.transpose();
                AA(m, m) = -zet / z;

                VectorXd solut = AA.fullPivLu().solve(bb);
                VectorXd dlam = solut.head(m);
                double dz = solut(m);

                VectorXd dx = -delx.cwiseQuotient(diagx) - (GG.transpose() * dlam).cwiseQuotient(diagx);
                VectorXd dy = -dely.cwiseQuotient(diagy) + dlam.cwiseQuotient(diagy);
                VectorXd dxsi = -xsi + epsvecn.cwiseQuotient(x - alfa) - xsi.cwiseProduct(dx).cwiseQuotient(x - alfa);
                VectorXd deta = -eta + epsvecn.cwiseQuotient(beta - x) + eta.cwiseProduct(dx).cwiseQuotient(beta - x);
                VectorXd dmu = -mu + epsvecm.cwiseQuotient(y) - mu.cwiseProduct(dy).cwiseQuotient(y);
                double   dzet = -zet + epsi / z - zet * dz / z;
                VectorXd ds = -s + epsvecm.cwiseQuotient(lam) - s.cwiseProduct(dlam).cwiseQuotient(lam);

                VectorXd xx(nxx);
                off = 0;
                xx.segment(off, m) = y;     off += m;
                xx(off++) = z;
                xx.segment(off, m) = lam;   off += m;
                xx.segment(off, n) = xsi;   off += n;
                xx.segment(off, n) = eta;   off += n;
                xx.segment(off, m) = mu;    off += m;
                xx(off++) = zet;
                xx.segment(off, m) = s;     off += m;

                if (off != nxx) {
                    throw std::runtime_error("subsolv: xx assembly size mismatch.");
                }

                VectorXd dxx(nxx);
                off = 0;
                dxx.segment(off, m) = dy;    off += m;
                dxx(off++) = dz;
                dxx.segment(off, m) = dlam;  off += m;
                dxx.segment(off, n) = dxsi;  off += n;
                dxx.segment(off, n) = deta;  off += n;
                dxx.segment(off, m) = dmu;   off += m;
                dxx(off++) = dzet;
                dxx.segment(off, m) = ds;    off += m;

                if (off != nxx) {
                    throw std::runtime_error("subsolv: dxx assembly size mismatch.");
                }

                double stmxx = (-1.01 * dxx.array() / xx.array()).maxCoeff();
                double stmalfa = (-1.01 * dx.array() / (x - alfa).array()).maxCoeff();
                double stmbeta = (1.01 * dx.array() / (beta - x).array()).maxCoeff();
                double stminv = std::max(1.0, std::max(stmxx, std::max(stmalfa, stmbeta)));
                double steg = 1.0 / stminv;

                VectorXd xold = x;
                VectorXd yold = y;
                VectorXd lamold = lam;
                VectorXd xsiold = xsi;
                VectorXd etaold = eta;
                VectorXd muold = mu;
                VectorXd sold = s;
                double zold = z;
                double zetold = zet;

                double resinew = 2.0 * residunorm;
                int itto = 0;

                while (resinew > residunorm && itto < 8) {
                    ++itto;

                    x = xold + steg * dx;
                    y = yold + steg * dy;
                    z = zold + steg * dz;
                    lam = lamold + steg * dlam;
                    xsi = xsiold + steg * dxsi;
                    eta = etaold + steg * deta;
                    mu = muold + steg * dmu;
                    zet = zetold + steg * dzet;
                    s = sold + steg * ds;

                    ux1 = upp - x;
                    xl1 = x - low;
                    ux2 = ux1.array().square().matrix();
                    xl2 = xl1.array().square().matrix();
                    uxinv1 = ux1.cwiseInverse();
                    xlinv1 = xl1.cwiseInverse();

                    plam = p0 + P.transpose() * lam;
                    qlam = q0 + Q.transpose() * lam;
                    gvec = P * uxinv1 + Q * xlinv1;
                    dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);

                    rex = dpsidx - xsi + eta;
                    rey = c + d.cwiseProduct(y) - mu - lam;
                    rez = a0 - zet - a.dot(lam);
                    relam = gvec - a * z - y + s - b;
                    rexsi = xsi.cwiseProduct(x - alfa) - epsvecn;
                    reeta = eta.cwiseProduct(beta - x) - epsvecn;
                    remu = mu.cwiseProduct(y) - epsvecm;
                    rezet = zet * z - epsi;
                    res = lam.cwiseProduct(s) - epsvecm;

                    VectorXd residuNew(nres);
                    off = 0;
                    residuNew.segment(off, n) = rex;    off += n;
                    residuNew.segment(off, m) = rey;    off += m;
                    residuNew(off++) = rez;
                    residuNew.segment(off, m) = relam;  off += m;
                    residuNew.segment(off, n) = rexsi;  off += n;
                    residuNew.segment(off, n) = reeta;  off += n;
                    residuNew.segment(off, m) = remu;   off += m;
                    residuNew(off++) = rezet;
                    residuNew.segment(off, m) = res;    off += m;

                    if (off != nres) {
                        throw std::runtime_error("subsolv: residuNew assembly size mismatch.");
                    }

                    resinew = residuNew.norm();
                    if (resinew > residunorm) {
                        steg *= 0.5;
                    }
                }

                VectorXd residuUpd(nres);
                off = 0;
                residuUpd.segment(off, n) = rex;    off += n;
                residuUpd.segment(off, m) = rey;    off += m;
                residuUpd(off++) = rez;
                residuUpd.segment(off, m) = relam;  off += m;
                residuUpd.segment(off, n) = rexsi;  off += n;
                residuUpd.segment(off, n) = reeta;  off += n;
                residuUpd.segment(off, m) = remu;   off += m;
                residuUpd(off++) = rezet;
                residuUpd.segment(off, m) = res;    off += m;

                if (off != nres) {
                    throw std::runtime_error("subsolv: residuUpd assembly size mismatch.");
                }

                residu = residuUpd;
                residunorm = residu.norm();
                residumax = residu.cwiseAbs().maxCoeff();
            }

            epsi *= 0.1;
        }

        MMAResult out;
        out.xmma = x;
        out.ymma = y;
        out.zmma = z;
        out.lam = lam;
        out.xsi = xsi;
        out.eta = eta;
        out.mu = mu;
        out.zet = zet;
        out.s = s;
        return out;
    }

    MMAResult mmasub(int m, int n, int iter,
        const VectorXd & xval, const VectorXd & xmin, const VectorXd & xmax,
        const VectorXd & xold1, const VectorXd & xold2,
        double f0val, const VectorXd & df0dx,
        const VectorXd & fval, const MatrixXd & dfdx,
        const VectorXd & low_in, const VectorXd & upp_in,
        double a0, const VectorXd & a, const VectorXd & c, const VectorXd & d) {
        (void)f0val;
        const double epsimin = 1e-4;
        const double raa0 = 1e-5;
        const double move = 0.2;
        const double albefa = 0.1;
        const double asyinit = 0.1;
        const double asyincr = 1.2;
        const double asydecr = 0.7;

        VectorXd eeen = VectorXd::Ones(n);
        VectorXd eeem = VectorXd::Ones(m);
        VectorXd zeron = VectorXd::Zero(n);

        VectorXd low = low_in;
        VectorXd upp = upp_in;

        if (iter < 3) {
            low = xval - asyinit * (xmax - xmin);
            upp = xval + asyinit * (xmax - xmin);
        }
        else {
            VectorXd zzz = (xval - xold1).cwiseProduct(xold1 - xold2);
            VectorXd factor = eeen;
            for (int i = 0; i < n; ++i) {
                if (zzz(i) > 0.0) factor(i) = asyincr;
                else if (zzz(i) < 0.0) factor(i) = asydecr;
            }
            low = xval - factor.cwiseProduct(xold1 - low);
            upp = xval + factor.cwiseProduct(upp - xold1);
            VectorXd lowmin = xval - 10.0 * (xmax - xmin);
            VectorXd lowmax = xval - 0.01 * (xmax - xmin);
            VectorXd uppmin = xval + 0.01 * (xmax - xmin);
            VectorXd uppmax = xval + 10.0 * (xmax - xmin);
            low = low.cwiseMax(lowmin).cwiseMin(lowmax);
            upp = upp.cwiseMin(uppmax).cwiseMax(uppmin);
        }

        VectorXd alfa = (low + albefa * (xval - low)).cwiseMax(xval - move * (xmax - xmin)).cwiseMax(xmin);
        VectorXd beta = (upp - albefa * (upp - xval)).cwiseMin(xval + move * (xmax - xmin)).cwiseMin(xmax);

        VectorXd xmami = (xmax - xmin).cwiseMax(1e-5 * eeen);
        VectorXd xmamiinv = xmami.cwiseInverse();
        VectorXd ux1 = upp - xval;
        VectorXd ux2 = ux1.array().square().matrix();
        VectorXd xl1 = xval - low;
        VectorXd xl2 = xl1.array().square().matrix();
        VectorXd uxinv = ux1.cwiseInverse();
        VectorXd xlinv = xl1.cwiseInverse();

        VectorXd p0 = df0dx.cwiseMax(0.0);
        VectorXd q0 = (-df0dx).cwiseMax(0.0);
        VectorXd pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv;
        p0 = (p0 + pq0).cwiseProduct(ux2);
        q0 = (q0 + pq0).cwiseProduct(xl2);

        MatrixXd P = dfdx.cwiseMax(0.0);
        MatrixXd Q = (-dfdx).cwiseMax(0.0);
        MatrixXd PQ = 0.001 * (P + Q);
        for (int i = 0; i < m; ++i) PQ.row(i) += raa0 * xmamiinv.transpose();
        P = P + PQ;
        Q = Q + PQ;
        P = P * ux2.asDiagonal();
        Q = Q * xl2.asDiagonal();
        VectorXd b = P * uxinv + Q * xlinv - fval;

        MMAResult out = subsolv_generic(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d);
        out.low = low;
        out.upp = upp;
        return out;
    }
