#ifndef _RBM_H_
#define _RBM_H_

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include "constants.h"
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
    typedef std::vector<std::reference_wrapper<mat>> vec_p;

    std::random_device rd;

    struct gama
    {
        long double g = 0;
        long double rate = gama_decrement_exponent;
        int *int_123 = nullptr;
        gama(double r = .01, int *n = nullptr)
        {
            g = r;
            rate = gama_decrement_exponent;
            int_123 = n;
        }
        auto operator*(mat m)
        {
            double t = g / pow((*int_123), rate);
            return t * m;
        }
        double out()
        {
            return g / pow((*int_123), rate);
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
        void shake()
        {
            W += 0.01 * (arma::randu(hid_node_num, row));
            a += 0.01 * (arma::randu(row, 1));
            b += 0.01 * (arma::randu(hid_node_num, 1));
        }
        void shake(gama g)
        {
            W += g * (arma::randu(hid_node_num, row));
            a += g * (arma::randu(row, 1));
            b += g * (arma::randu(hid_node_num, 1));
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
                   { i = tanh(activation_function(i)) * activation_function_derivative(i); });
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
    bool inf_check(long double a)
    {
        if (std::isnan(a) || std::isinf(a))
        {
            std::cout << "the result was inf/nan" << "\n\n\n";
            return true;
        }
        return false;
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
                   { m = log((cosh(activation_function(m)))); });
        for (size_t i = 0; i < hid_node_num; i++)
        {
            cosh_theta = log(2) + cosh_theta + (m(i, 0));
        }
        psi = (cosh_theta) + (sig_i_a_i);
        inf_check(psi);
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

    // In namespace pj

    // Calculates log(psi(S')) - log(psi(S)) efficiently for a single spin flip at index 'k'
    long double log_psi_diff(int k, const visible_layer &VL, const weights &w)
    {
        const mat &S = VL.S;
        const mat &W = w.W;
        const mat &a = w.a;
        const mat &b = w.b;

        long double delta_log_psi = -2.0 * a(k, 0) * S(k, 0);

        mat theta = w.b + w.W * S; // Original theta

        for (size_t j = 0; j < hid_node_num; ++j)
        {
            double theta_j_prime = theta(j, 0) - 2.0 * W(j, k) * S(k, 0);
            delta_log_psi += log(cosh(activation_function(theta_j_prime))) - log(cosh(activation_function(theta(j, 0))));
        }
        return delta_log_psi;
    }

    long double log_psi_sum(int k, const visible_layer &VL, const weights &w)
    {
        const mat &S = VL.S;
        const mat &W = w.W;
        const mat &a = w.a;
        const mat &b = w.b;

        long double delta_log_psi = arma::as_scalar(S.t() * a) - 2.0 * a(k, 0) * S(k, 0);

        mat theta = w.b + w.W * S; // Original theta

        for (size_t j = 0; j < hid_node_num; ++j)
        {
            double theta_j_prime = theta(j, 0) - 2.0 * W(j, k) * S(k, 0);
            delta_log_psi += log(cosh(activation_function(theta_j_prime))) + log(cosh(activation_function(theta(j, 0))));
        }
        return delta_log_psi;
    }

    long double p_ratio_fast(int k_flipped, const visible_layer &VL, const weights &w)
    {
        return exp(log_psi_diff(k_flipped, VL, w));
    }

    long double p_prod(int k_flipped, const visible_layer &vl, const weights &w)
    {
        return exp(log_psi_sum(k_flipped, vl, w));
    }

    // Update E_loc to use this new function
    long double E_loc_fast(visible_layer VL, const weights &W)
    {
        long double e_loc_val = 0;
        // Term for interactions: J * sum(sig_z(i)*sig_z(i+1))
        for (size_t i = 0; i < row - 1; i++)
        {
            e_loc_val += -J * VL.S(i, 0) * VL.S((i + 1) % row, 0);
        }

        // Term for transverse field: H * sum(sig_x)
        for (size_t i = 0; i < row; i++)
        {
            e_loc_val += -H * p_ratio_fast(i, VL, W);
        }
        return e_loc_val;
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
            e_loc += E_loc_fast(VL2, W);
            // vl3 = VL2;
            // }
        }
        return e_loc / itt_no;
    }

    long double magnetization_x(const visible_layer &vl, const weights &w)
    {
        long double m_x = 0;
        visible_layer vl_m = vl;
        for (size_t i = 0; i < row - 1; i++)
        {
            m_x += log_psi_sum(i, vl, w);
        }
        return m_x;
    }

    long double magnetization_avg_x(visible_layer &vl, const weights &w, int N = itt_value)
    {
        long double avg_m = 0;
        for (size_t i = 0; i < N; i++)
        {
            vl = sampler(vl, w);
            avg_m += magnetization_x(vl, w);
        }
        return avg_m / N;
    }

    // thhis function keeps E_loc_avg in memory untill weights or VL is changed;

    void O_init(vector<mat> &O, vector<mat> &OT, vector<mat> &OT_O, vector<mat> &E_OT, const visible_layer &vl, const weights &w,
                vector<function<mat(const visible_layer, const weights &)>> matrix_maker,
                vector<function<visible_layer(const visible_layer, const weights &, std::random_device &)>> sampler_function)
    {
        visible_layer vl2 = vl;
        for (size_t j = 0; j < 3; j++)
        {
            O.push_back(matrix_maker[j](sampler_function[j](vl2, w, rd), w));
            OT.push_back(matrix_maker[j](sampler_function[j](vl2, w, rd), w).t());
            OT_O.push_back(matrix_maker[j](sampler_function[j](vl2, w, rd), w).t() *
                           matrix_maker[j](sampler_function[j](vl2, w, rd), w));
            E_OT.push_back((E_loc(vl2, w)) * matrix_maker[j](sampler_function[j](vl2, w, rd), w).t());
        }
    }

    void O_update(mat &O, mat &OT, mat &OT_O, mat &E_OT, visible_layer &vl, const weights &w, function<mat(const visible_layer, const weights &)> matrix_maker,
                  function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler_function, int N = itt_value)
    {
        vl = sampler_function(vl, w, rd);
        O += matrix_maker(sampler_function(vl, w, rd), w);
        OT += matrix_maker(sampler_function(vl, w, rd), w).t();
        OT_O += matrix_maker(sampler_function(vl, w, rd), w).t() *
                matrix_maker(sampler_function(vl, w, rd), w);
        E_OT += (E_loc(sampler_function(vl, w, rd), w)) * matrix_maker(sampler_function(vl, w, rd), w).t();
    }

    void O_averager(vector<mat> &O, vector<mat> &OT, vector<mat> &OT_O, vector<mat> &E_OT, visible_layer &vl, const weights &w,
                    vector<function<mat(const visible_layer, const weights &)>> matrix_maker,
                    vector<function<visible_layer(const visible_layer, const weights &, std::random_device &)>> sampler_function, int N = itt_value)
    {
        for (size_t j = 0; j < N; j++)
        {
            for (size_t i = 0; i < 3; i++)
            {
                O_update(O[i], OT[i], OT_O[i], E_OT[i], vl, w, matrix_maker[i], sampler_function[i]);
            }
        }
    }

    vector<mat> inv_S_F(visible_layer &vl, const weights &w, vector<function<visible_layer(const visible_layer, const weights &, std::random_device &)>> sampler_function,
                        vector<function<mat(const visible_layer, const weights &)>> matrix_maker, int N = itt_value)
    // w_update eye = hidden_node_num
    // eye_num is for the eye matrix for a_update = row
    // b_update eye = hidden_node_num
    {
        vector<int> eye_num;
        eye_num.push_back(hid_node_num);
        eye_num.push_back(row);
        eye_num.push_back(hid_node_num);

        vector<mat> O, OT_O, OT, E_OT,
            S, F, a_n, i, m; // = arma::eye(eye_num, eye_num);

        for (size_t j = 0; j < 3; j++)
        {
            i.push_back(arma::eye(eye_num[j], eye_num[j]));
        }

        static long double lamda = pow(10, 2), a = 100;
        a = a * 0.9;
        lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);

        O_init(O, OT, OT_O, E_OT, vl, w, matrix_maker, sampler_function);
        O_averager(O, OT, OT_O, E_OT, vl, w, matrix_maker, sampler_function);

        long double e_loc = E_loc_avg(vl, w);

        for (size_t j = 0; j < 3; j++)
        {
            S.push_back((OT_O[j] / N) - ((OT[j] * O[j] / pow(N, 2))));
            S[j] = S[j] + lamda * arma::diagmat(S[j]);
            // S = S + lamda * i;
            F.push_back((E_OT[j] / N) - (e_loc)*OT[j] / N);
            m.push_back((arma::pinv(S[j])) * F[j]);
        }

        return m;
    }

    weights W_update_chooser(visible_layer &vl, const weights &wei, int & n)
    {
        // static int n = 1;
        double value = E_loc_avg(vl,wei);
        weights w=wei,w2=wei;
        if(check_mulitple_vales_of_update)
        {
            for (size_t t = 0; t < no_of_mulitple_vales_of_update; t++)
            {          
                vector<function<visible_layer(const visible_layer, const weights &, std::random_device &)>> sampler_vector;
                for (size_t i = 0; i < 3; i++)
                {
                    sampler_vector.push_back(sampler);
                }
                
                
                vector<function<mat(const visible_layer, const weights &)>> matrix_maker;
                matrix_maker.push_back(vis_cross_tanh);
                matrix_maker.push_back(identity_vis_lay);
                matrix_maker.push_back(tanh_matrix);
                
                static gama g(gama_init_value, &n);
                
                vector<mat> W_update = inv_S_F(vl, w, sampler_vector, matrix_maker);
                
                (w.W) -= g * W_update[0];
                (w.a) -= g * W_update[1] * pow(10, -2);
                (w.b) -= g * W_update[2] * pow(10, -2);
                double a=E_loc_avg(vl,w); 
                if(a<value)
                {
                    w2=w;
                    value=a;
                }
                w=wei;
            }
            return w2;
        }
        else
        {
            vector<function<visible_layer(const visible_layer, const weights &, std::random_device &)>> sampler_vector;
                for (size_t i = 0; i < 3; i++)
                {
                    sampler_vector.push_back(sampler);
                }
                
                
                vector<function<mat(const visible_layer, const weights &)>> matrix_maker;
                matrix_maker.push_back(vis_cross_tanh);
                matrix_maker.push_back(identity_vis_lay);
                matrix_maker.push_back(tanh_matrix);
                
                static gama g(gama_init_value, &n);
                
                vector<mat> W_update = inv_S_F(vl, w, sampler_vector, matrix_maker);
                
                (w.W) -= g * W_update[0];
                (w.a) -= g * W_update[1] * pow(10, -2);
                (w.b) -= g * W_update[2] * pow(10, -2);
                return w;
        }
    }

    double W_update(visible_layer &vl, weights &w)
    {
        static int n = 1;
        w=W_update_chooser(vl,w,n);
        // vector<function<visible_layer(const visible_layer, const weights &, std::random_device &)>> sampler_vector;
        // for (size_t i = 0; i < 3; i++)
        // {
        //     sampler_vector.push_back(sampler);
        // }


        // vector<function<mat(const visible_layer, const weights &)>> matrix_maker;
        // matrix_maker.push_back(vis_cross_tanh);
        // matrix_maker.push_back(identity_vis_lay);
        // matrix_maker.push_back(tanh_matrix);

        gama g(gama_init_value, &n);

        // vector<mat> W_update = inv_S_F(vl, w, sampler_vector, matrix_maker);

        // (w.W) -= g * W_update[0];
        // (w.a) -= g * W_update[1] * pow(10, -2);
        // (w.b) -= g * W_update[2] * pow(10, -2);

        n++;
        // arma::norm(w.W);
        // g.update();
        return g.out();
    }
}

#endif