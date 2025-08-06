#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
#include <matplot/matplot.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <set>
#include <complex>
#include <unordered_map>
#include <random>
#include <stdexcept>

/* this will used the functional approach to make the
RBM
there will exist a struct of wieghts W and then this struct of wieghts will
we used to calculate the psi(s)*/
namespace pj
{
    typedef arma::cx_mat cmat;
    typedef arma::Mat<double> mat;
    typedef std::complex<long double> dclx;

    // defines the paramemeter of the sim
    int col = 1;
    int row = 10;
    int alpha = 2;
    int hid_node_num = alpha * row;
    double sigmoid(double x) { return ((1 / (1 + exp(0.1 * x))) + pow(10, -3)); }
    long double gama()
    {
        static double n = 0.1;
        return (n > pow(10, -4)) ? (n) : (pow(10, -4));
    }

    // parameters of the hamiltonian
    long double h = 0.5;
    long double j = 1;
    std::random_device rd;

    // debuging

    const int dim = pow(2, row);
    struct weights
    {
        mat W;
        mat a;
        mat b;
        weights()
        {
            W = 0 * arma::randu(hid_node_num, row);
            a = 0.1 * arma::randu(row, 1);
            b = 0.1 * arma::randu(hid_node_num, 1);
        }
        void operator/(double n)
        {
            W = W / n;
            a = a / n;
            b = b / n;
        }

        bool are_equal(const weights &w)
        {
            bool m = true;
            if ((w.W.n_cols != W.n_cols) && (w.W.n_rows != W.n_rows) && (w.a.n_cols != a.n_cols) && (w.a.n_rows != a.n_rows) && (w.b.n_rows != b.n_rows))
                return false;
            for (size_t i = 0; i < w.W.n_cols; i++)
            {
                for (size_t j = 0; j < w.W.n_rows; j++)
                {
                    m = m * (w.W(j, i) == W(j, i));
                }
                if (!m)
                    return false;
            }
            for (size_t i = 0; i < w.a.n_cols; i++)
            {
                for (size_t j = 0; j < w.a.n_rows; j++)
                {
                    m = m * (w.a(j, i) == a(j, i));
                }
                if (!m)
                    return false;
            }

            for (size_t i = 0; i < w.b.n_cols; i++)
            {
                for (size_t j = 0; j < w.b.n_rows; j++)
                {
                    m = m * (w.b(j, i) == b(j, i));
                }
            }
            return m;
        }
    };

    struct visible_layer
    {
        mat S = arma::randu(row, 1);
        visible_layer()
        {
            S.for_each([](mat::elem_type &m)
                       { (m > 0.5) ? (m = -1) : (m = 1); });
        }
        void flip(int i)
        {

            S(i, 0) = -S(i, 0);
        }
        bool are_equal(const visible_layer &vl)
        {
            bool m = true;
            if ((vl.S.n_rows != S.n_rows))
                return false;
            for (size_t i = 0; i < S.n_rows; i++)
            {

                m = m * (vl.S(i, 0) == S(i, 0));
            }
            return m;
        }
    };

    // these are small function which take  visible layer as an input and convert it into other matricies that are to be use in
    // the program some where else, their names are pretty self explainatory
    mat theta_matrix(const visible_layer &VL, const weights &w)
    {
        mat M(hid_node_num, 1);
        M = w.b + w.W * VL.S;
        return M;
    }
    mat tanh_matrix(const visible_layer &vl, const weights &w)
    {
        mat b = theta_matrix(vl, w);
        b.for_each([](auto &i)
                   { i = tanh(i); });
        return b.t();
    }
    mat identity_vis_lay(visible_layer vl, const weights &w)
    {
        return vl.S.t();
    }
    mat vis_cross_tanh(visible_layer vl, const weights &w)
    {
        return vl.S * tanh_matrix(vl, w);
    }

    // this functions checks for invalid numbers taht might creep up in the program
    void bruh(long double a)
    {
        if (std::isnan(a))
            throw runtime_error("nan");
        if (std::isinf(a))
            throw runtime_error("inf");
    }

    long double psi(visible_layer VL, const weights &WEI) // to calculate the probability psi(s) for a given weights and visible layers
    {
        long double psi = 0; // initialization of psi

        // sigmai* a(i) implimnetation
        long double sig_i_a_i = 0;
        sig_i_a_i = arma::as_scalar(VL.S.t() * WEI.a);
        if (std::isnan(sig_i_a_i))
        {
            std::cout<<"well we hit nan\n";
            sig_i_a_i = 0;
        }

        long double cosh_theta = 0;
        mat m = theta_matrix(VL, WEI);
        m.for_each([](auto &m)
                   { m = log(cosh(m)); });
        for (size_t i = 0; i < hid_node_num; i++)
        {
            cosh_theta = log(2) + cosh_theta + (m(i, 0));
        }
        bruh(cosh_theta);
        bruh(sig_i_a_i);
        psi = (cosh_theta) + (sig_i_a_i);
        return (psi);
    }

    long double p_ratio(visible_layer VL, visible_layer VL2, const weights &w)
    {
        long double a = psi(VL, w), b = psi(VL2, w);
        bruh(exp(a - b));
        return exp(a - b);
    }

    visible_layer sampler(visible_layer VL, const weights &w, std::random_device &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist(0, VL.S.n_rows - 1);
        std::uniform_real_distribution<double> realdist(0, 1);
        int n = dist(rd);
        visible_layer vl = VL;
        vl.flip(n);
        if (pow(p_ratio(vl, VL, w), 2) > 1)
            return vl;
        else
        {
            double r = realdist(rd);
            if (pow(p_ratio(vl, VL, w), 2) > r)
                return vl;
        }
        return VL;
    }

    long double E_loc(visible_layer VL, const weights &W)
    {

        // the hamiltonian is h*sum(sig_x)+ j*sum(sig_z(i)*sig_z(i+1))
        long double E_loc = 0;
        visible_layer m = VL;
        for (size_t i = 0; i < row - 1; i++)
        {
            m.flip(i);
            E_loc += -j * VL.S(i % row, 0) * VL.S((i + 1) % row) - h * p_ratio(m, VL, W);
            m = VL;
        }

        m.flip(row - 1);
        E_loc += -h * p_ratio(m, VL, W);
        return E_loc;
    }

    long double E_loc_avg(const visible_layer VL, const weights &W, int itt_no = 1000, std::random_device &rd = pj::rd)
    {
        long double e_loc = E_loc(VL, W);
        visible_layer VL2 = VL;
        for (size_t i = 0; i < itt_no - 1; i++)
        {
            VL2 = sampler(VL2, W);
            e_loc += E_loc(VL2, W);
        }
        return e_loc / (itt_no);
    }

    // thhis function keeps E_loc_avg in memory untill weights or VL is changed;
    long double E_loc_avg_eff(const visible_layer VL, const weights &W)
    {
        static visible_layer vl = VL;
        static weights w = W;
        static long double E = E_loc_avg(VL, W);
        if (w.are_equal(W) && vl.are_equal(VL))
        {
            return E;
        }
        else
        {
            vl = VL;
            w = W;
            E = E_loc_avg(vl, W);
            return E;
        }
        throw std::runtime_error("should not reach the end of function E_loc_avg_eff");
    }

    mat inv_S_F(visible_layer vl, const weights &w, function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler_function,
                function<mat(const visible_layer, const weights &)> matrix_maker, int eye_num, int N = 500)
    // eye_num is for the eye matrix for a_update = row
    // b_update eye = hidden_node_num
    // w_update eye = hidden_node_num
    {
        mat O, OT_O, OT, O_O, E_OT,
            S, F, a_n, i = arma::eye(eye_num, eye_num);

        static double lamda = pow(10, 2), a = 100;
        a = a * 0.9995;
        lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);

        visible_layer vl2 = vl;
        O = matrix_maker(sampler_function(vl2, w, rd), w);
        OT = matrix_maker(sampler_function(vl2, w, rd), w).t();
        OT_O = matrix_maker(sampler_function(vl2, w, rd), w).t() *
               matrix_maker(sampler_function(vl2, w, rd), w);
        E_OT = E_loc(vl2, w) * matrix_maker(sampler_function(vl2, w, rd), w).t();

        for (size_t j = 1; j < N; j++)
        {
            O += matrix_maker(sampler_function(vl2, w, rd), w);
            OT += matrix_maker(sampler_function(vl2, w, rd), w).t();
            OT_O += matrix_maker(sampler_function(vl2, w, rd), w).t() *
            matrix_maker(sampler_function(vl2, w, rd), w);
            E_OT += E_loc(sampler_function(vl2, w, rd), w) * matrix_maker(sampler_function(vl2, w, rd), w).t();
            vl2 = sampler_function(vl, w, rd);
        }

        long double e_loc = E_loc_avg_eff(vl, w);
        S = (OT_O / N) - ((OT * O / pow(N, 2)));
        S = S + lamda * arma::diagmat(S);
        F = (E_OT / N) - e_loc * OT / N;
        mat m = ((arma::pinv(S)) * F);
        return m / arma::norm(m);
    }

    mat a_update(visible_layer vl, const weights &w, int N = 1000)
    {
        mat m = w.a - gama() * inv_S_F(vl, w, &sampler, &identity_vis_lay, row);
        return m;
    }

    //! mark this

    mat w_update(visible_layer vl, const weights &w, int N = 1000)
    {
        mat m = w.W - gama() * inv_S_F(vl, w, &sampler, &vis_cross_tanh, hid_node_num);

        return m;
    }

    mat b_update(visible_layer vl, const weights &w, int N = 1000)
    {
        mat m = w.b - gama() * inv_S_F(vl, w, &sampler, &tanh_matrix, hid_node_num);
        return m;
    }

}

#endif