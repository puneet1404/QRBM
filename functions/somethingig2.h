#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include"constants.h"
// #include <ginac/ginac.h>

/* this will used the functional approach to make the
RBM
there will exist a struct of wieghts W and then this struct of wieghts will
we used to calculate the psi(s)*/
namespace pj
{
    // namespace g = GiNaC;
    // typedef g::long long;
    typedef arma::cx_mat cmat;
    typedef arma::Mat<double> mat;
    typedef std::complex<double> dclx;

    // defines the paramemeter of the sim
    // int col = 1;
    // int row = 6;
    // int alpha = 1;
    // int hid_node_num = alpha * row;
    // long double H = 0.5;
    // long double J = 1;

    // double gama_init_value = .01;
    // double gama_decrement_exponent=1;
    // double itt_value = 1000;
    
    
    // long long double gama()
    // {
    //     static long double n = .1;
    //     n *= (.95);
    //     return (n > pow(10, -4)) ? (n) : (pow(10, -4));
    // }

    // parameters of the hamiltonian
    std::random_device rd;

    struct gama
    {
        long double g = 0;
        long double rate = gama_decrement_exponent;
        int *int_123;
        gama(double r = .01, int *n = nullptr)
        {
            g = r;
            rate = gama_decrement_exponent;
            int_123 = n;
        }
        auto operator*(mat m)
        {
            // static double thresh_hold= 0.05;
            // mat t = g*m;
            // if ( thresh_hold<g)
            // {
            //     g*=(g<pow(10,-5)?(1):(rate));
            //     thresh_hold*=(thresh_hold<pow(10,-4)?(1):(rate));
            //     return g * m;
            // }
            // if (arma::norm(t)>thresh_hold)
            // {
            //     thresh_hold*=rate;
            //     // cout<<"sad to be here \n";
            //     return (thresh_hold/arma::norm(m))*m;
            // }
            // std::cout<<*int_123<<"\n";
            double t = g / pow((*int_123), rate);
            // return ((t > pow(10, -7)) ? (t) : (pow(10, -7))) * m;
            return t * m;
        }
        double out()
        {
            return g / pow((*int_123), rate);
            // g=(g > pow(10, -4)) ? (g) : (pow(10, -4));
        }
    };

    struct weights
    {
        mat W;
        mat a;
        mat b;
        weights()
        {
            W = 0.1 * arma::randu(hid_node_num, row);
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
        void normalize()
        {
            W = W / arma::norm(W);
            a = a / arma::norm(a);
            b = b / arma::norm(b);
        }
        void shake(gama g)
        {
            W += g* (arma::randu(hid_node_num, row));
            a += g* (arma::randu(row, 1));
            b += g* (arma::randu(hid_node_num, 1));
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
        return (vl.S * tanh_matrix(vl, w));
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

        long double cosh_theta = 0;
        mat m = theta_matrix(VL, WEI);
        m.for_each([](auto &m)
                   { m = log(cosh(m)); });
        for (size_t i = 0; i < hid_node_num; i++)
        {
            cosh_theta = log(2) + cosh_theta + (m(i, 0));
        }
        psi = (cosh_theta) + (sig_i_a_i);
        bruh(psi);
        return (psi);
    }

    long double p_ratio(visible_layer VL, visible_layer VL2, const weights &w)
    {
        long double a = psi(VL, w), b = psi(VL2, w);
        return exp(a - b);
    }

    visible_layer sampler(visible_layer VL, const weights &w, std::random_device &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist(0, VL.S.n_rows - 1);
        std::uniform_real_distribution<long double> realdist(0, 1);
        visible_layer vl = VL;
        vl.flip(dist(rd));
        if ((pow(p_ratio(vl, VL, w), 2) > realdist(rd)) || p_ratio(vl, VL, w) > 1)
            return vl;
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
            E_loc += -J * VL.S(i % row, 0) * VL.S((i + 1) % row) - H * p_ratio(m, VL, W);
            m = VL;
        }

        m.flip(row - 1);
        E_loc += -H * p_ratio(m, VL, W);
        return E_loc;
    }

    long double E_loc_avg(const visible_layer VL, const weights &W, int itt_no = itt_value, std::random_device &rd = pj::rd)
    {
        long double e_loc = E_loc(VL, W);
        visible_layer VL2 = VL, vl3 = VL2;
        for (size_t i = 0; i < itt_no - 1; i++)
        {
            VL2 = sampler(VL2, W);
            e_loc += E_loc(VL2, W);
            // vl3 = VL2;
            // }
        }
        return e_loc / itt_no;
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

    mat inv_S_F(visible_layer &vl, const weights &w, function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler_function,
                function<mat(const visible_layer, const weights &)> matrix_maker, int eye_num, int N = itt_value)
    // eye_num is for the eye matrix for a_update = row
    // b_update eye = hidden_node_num
    // w_update eye = hidden_node_num
    {
        mat O, OT_O, OT, E_OT,
            S, F, a_n, i = arma::eye(eye_num, eye_num);

        static long double lamda = pow(10, 2), a = 100;
        a = a * 0.99;
        lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);

        visible_layer vl2 = vl, vl3 = vl2;
        O = matrix_maker(sampler_function(vl2, w, rd), w);
        OT = matrix_maker(sampler_function(vl2, w, rd), w).t();
        OT_O = matrix_maker(sampler_function(vl2, w, rd), w).t() *
               matrix_maker(sampler_function(vl2, w, rd), w);
        E_OT = (E_loc(vl2, w)) * matrix_maker(sampler_function(vl2, w, rd), w).t();

        for (size_t j = 1; j < N - 1; j++)
        {
            vl2 = sampler_function(vl, w, rd);
            // if (!vl3.are_equal(vl2))
            // {

            O += matrix_maker(vl2, w);
            OT += matrix_maker(vl2, w).t();
            OT_O += matrix_maker(vl2, w).t() *
                    matrix_maker(vl2, w);
            E_OT += (E_loc(vl2, w)) * matrix_maker(vl2, w).t();
            // vl3 = vl2;
            // }
        }
        vl = vl2;

        long double e_loc = E_loc_avg_eff(vl, w);
        S = (OT_O / N) - ((OT * O / pow(N, 2)));
        S = S + lamda * arma::diagmat(S);
        // S = S + lamda * i;
        F = (E_OT / N) - (e_loc)*OT / N;
        mat m = ((arma::pinv(S)) * F);
        return m;
    }

    // mat a_update(visible_layer vl, const weights &w)
    // {
    //     mat m = w.a - gama() * inv_S_F(vl, w, &sampler, &identity_vis_lay, row);
    //     return m;
    // }
    // mat w_update(visible_layer vl, const weights &w)
    // {
    //     mat m = w.W - gama() * inv_S_F(vl, w, &sampler, &vis_cross_tanh, hid_node_num);
    //
    //     return m;
    // }
    // mat b_update(visible_layer vl, const weights &w)
    // {
    //     mat m = w.b - gama() * inv_S_F(vl, w, &sampler, &tanh_matrix, hid_node_num);
    //     return m;
    // }
    double W_update(visible_layer &vl, weights &w)
    {
        static int n = 1;
        static gama g(gama_init_value, &n);
        mat W = inv_S_F(vl, w, &sampler, &vis_cross_tanh, hid_node_num),
            a = inv_S_F(vl, w, &sampler, &identity_vis_lay, row),
            b = inv_S_F(vl, w, &sampler, &tanh_matrix, hid_node_num);
        w.W = w.W - g * W;
        w.a = w.a - g * a * pow(10, -3);
        w.b = w.b - g * b * pow(10, -3);
        n++;
        // arma::norm(w.W);
        // g.update();
        return g.out();
    }
}

#endif