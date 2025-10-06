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
#include <set>
#include "constants.h"
// #include <ginac/ginac.h>

/* this will used the functional approach to make the
RBM
there will exist a struct of wieghts W and then this struct of wieghts will
we used to calculate the psi(s)*/
using namespace std;
using namespace std::complex_literals;
namespace pj
{
    // namespace g = GiNaC;
    // typedef g::long long;
    typedef arma::cx_mat mat;
    typedef arma::Mat<double> dmat;
    typedef std::complex<double> dclx;
    // typedef std::complex_literals::il
    // typedef std::complex_literals::1i img;
    // typedef std::vector<std::reference_wrapper<mat>> vec_p;

    std::random_device rd_seed;
    std::mt19937_64 rd{rd_seed()};
    struct gama
    {
        long double g = 0;
        double init_value;
        long double rate = gama_decrement_exponent;
        int *int_123 = nullptr;
        double t;
        bool flag = true;
        gama(double r = .01, int *n = nullptr)
        {
            g = r;
            init_value = r;
            rate = gama_decrement_exponent;
            int_123 = n;
            t = g;
        }
        mat operator*(mat m)
        {
            // if ( g<.01)
            // {

            //     if (*int_123%1000==0)
            //     g=g+0.001;
            //     // return ((g+0.001)*m);
                
            //     // if (*int_123 > 600)
            //     //     return (g * m) / pow(10, 3);
                
            //     // if (*int_123 > 400)
            //     //     return (g * m) / pow(10, 2);
                
            //     // if (*int_123 > 100)
            //     //     return (g * m) / pow(10, 2); // pow(10, 1);
            //     cout<<"\ng\t\t="<<g<<endl;
            // }
            g*= pow(rate,1.0/3000.0);
            cout<<g<<endl;
            return ((g)*m);
        }
        double out()
        {
            return g; // pow(10, int(log10(*int_123)) - 1);
        }
    };

    struct weights
    {
        mat W;
        mat a;
        mat b;
        weights()
        {
            W = arma::randn(hid_node_num, row, arma::distr_param(mean, sd)) + 1i * ((pj::H >= 2) ? (arma::zeros(hid_node_num, row)) : (arma::randn(hid_node_num, row, arma::distr_param(mean, sd))));
            a = arma::randn(row, 1, arma::distr_param(mean, sd)) + 1i * ((pj::H >= 1) ? (arma::zeros(row, 1)) : (arma::randn(row, 1, arma::distr_param(mean, sd))));
            b = arma::randn(hid_node_num, 1, arma::distr_param(mean, sd)) + 1i * ((pj::H >= 2) ? (arma::zeros(hid_node_num, 1)) : (arma::randn(hid_node_num, 1, arma::distr_param(mean, sd))));
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
    };

    struct visible_layer
    {
        dmat S = arma::randu(row, 1);
        visible_layer()
        {
            S.for_each([](dmat::elem_type &m)
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
        int to_int()
        {
            unsigned long int b = 0;
            for (size_t i = 1; i < row + 1; i++)
            {
                b = b << 1;
                b = (S(row - i, 0) == -1) ? ((b + 1)) : (b);
            }
            return b;
        }
        void to_S(int n)
        {
            for (size_t i = 0; i < row; i++)
            {
                S(i) = (n % 2 == 0) ? (1) : (-1);
                n = n >> 1;
            }
        }
        void random()
        {
            S = arma::randu(row, 1);
            S.for_each([](dmat::elem_type &m)
                       { (m > 0.5) ? (m = -1) : (m = 1); });
        }
    };

    struct state
    {
        weights w;
        visible_layer vl;
        gama g;
        state(string g)
        {
            fstream file(g);
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
        return   arma::tanh(theta_matrix(vl, w)).t();
    }
    mat identity_vis_lay(visible_layer vl, const weights &w)
    {
        return   vl.S.t() + 1i * arma::zeros(arma::size(vl.S.t()));
    }
    mat vis_cross_tanh(visible_layer vl, const weights &w)
    {
        return (  vl.S * tanh_matrix(vl, w));
    }

    // this functions checks for invalid numbers taht might creep up in the program
    bool inf_check(long double a)
    {
        if (std::isnan(a) || std::isinf(a))
        {
            std::cout << "the result was inf/nan" << "\n\n\n";
            throw std::runtime_error("fuck you ");
        }
        return false;
    }

    dclx psi(visible_layer VL, const weights &WEI) // to calculate the probability psi(s) for a given weights and visible layers
    {
        dclx psi = 0; // initialization of psi

        // sigmai* a(i) implimnetation
        dclx sig_i_a_i = 0;
        sig_i_a_i = arma::accu(VL.S.t() * WEI.a);

        dclx cosh_theta = 0;
        mat m = theta_matrix(VL, WEI);
        // m.for_each([](auto &m)
        //            { m = log((cosh((m)))); });
        m =arma::log(2*arma::cosh(m));
        cosh_theta =arma::accu(m);
        // for (size_t i = 0; i < hid_node_num; i++)
        // {
        //     cosh_theta = log(2) + cosh_theta + (m(i, 0));
        // }
        psi = (cosh_theta) + (sig_i_a_i);
        inf_check(psi.real());
        inf_check(psi.imag());
        return (psi  );
    }

    dclx p_ratio(visible_layer VL, visible_layer VL2, const weights &w)
    {
        return exp(psi(VL, w) - psi(VL2, w));
    }
    dclx p_ratio(const int n , const visible_layer &vl, const weights& w)
    {
        visible_layer m =vl;
        m.flip(n);
        return p_ratio(m,vl,w);
        
    }
    visible_layer sampler_mp(visible_layer VL, const weights &w, std::mt19937_64 &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist(0, VL.S.n_rows - 1);
        std::uniform_real_distribution<long double> realdist(0, 1);
        visible_layer vl = VL;
        // for (size_t i = 0; i < spin_flip_num; i++)
        // {
        vl.flip(dist(rd));
        // }
        // cout<<vl.S<<"\n";
        if ( norm(p_ratio(vl, VL, w)) > 1||(pow(abs(p_ratio(vl, VL, w)), 2) > realdist(rd)))
            VL = vl;

        return VL;
    }

    visible_layer sampler_rw(visible_layer VL, const weights &w, std::mt19937_64 &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist(0, VL.S.n_rows - 1);
        visible_layer vl = VL;
        vl.flip(dist(rd));
        return VL;
    }

    // Calculates log(psi(S')) - log(psi(S)) efficiently for a single spin flip at index 'k'

    long double E_loc(visible_layer vl, const weights &W)
    {

        // the hamiltonian is h*sum(sig_x)+ j*sum(sig_z(i)*sig_z(i+1))
        long double E_loc = 0;
        visible_layer m = vl;
        for (size_t i = 0; i < row; i++)
        {
            E_loc += -J * vl.S(i%row)*vl.S((i+1)%row); // - H * p_ratio(m, vl, W);
        }
        for (size_t i = 0; i < row; i++)
        {
            m.flip(i);
            E_loc += -H * real(p_ratio(m, vl, W));
            m = vl;
        }
        // E_loc += -H*arma::accu(vl.S);
        return E_loc;
    }

    long double E_loc_avg(const visible_layer VL, const weights &W, int itt_no = itt_value, std::mt19937_64 &rd = pj::rd)
    {
        uniform_real_distribution<double> realdist(0, 1);
        long double e_loc = 0;
        visible_layer vl2 = VL, vl3 = vl2;
        int a = 50;
        int n = itt_no / a;
        for (size_t i = 0; i < a; i++)
        {
            for (size_t i = 0; i < n; i++)
            {
                vl2 = sampler_mp(vl2, W);
                e_loc += E_loc(vl2, W);
            }
        }

        return e_loc / (n * a);
    }

    void O_init(vector<mat> &O, vector<mat> &OT, vector<mat> &OT_O, vector<mat> &E_OT, const visible_layer &vl, const weights &w,
                vector<function<mat(const visible_layer, const weights &)>> matrix_maker,
                vector<function<visible_layer(const visible_layer, const weights &, std::mt19937_64 &)>> sampler_function)
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
                  function<visible_layer(const visible_layer, const weights &, std::mt19937_64 &)> sampler_function, int N = itt_value)
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
                    vector<function<visible_layer(const visible_layer, const weights &, std::mt19937_64 &)>> sampler_function, int N = itt_value)
    {
        for (size_t j = 0; j < N; j++)
        {
            for (size_t i = 0; i < 3; i++)
            {
                O_update(O[i], OT[i], OT_O[i], E_OT[i], vl, w, matrix_maker[i], sampler_function[i]);
            }
        }
    }

    vector<mat> inv_S_F(visible_layer &vl, const weights &w, vector<function<visible_layer(const visible_layer, const weights &, std::mt19937_64 &)>> sampler_function,
                        vector<function<mat(const visible_layer, const weights &)>> matrix_maker, int N = itt_value)
    {
        vector<int> eye_num;
        eye_num.push_back(hid_node_num);
        eye_num.push_back(row);
        eye_num.push_back(hid_node_num);

        vector<mat> O, OT_O, OT, E_OT,
            S, F, a_n, m; // = arma::eye(eye_num, eye_num);

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
        // double max_norm = 0;
        for (size_t i = 0; i < 3; i++)
        {
            if (arma::norm(m[i]) == 0)
            {
                continue;
            }
            else
            {

                m[i]/=arma::norm(m[i]);
            }
        }
        return m;
    }

    weights W_update_chooser(visible_layer &vl, const weights &wei, int &n)
    {
        // static int n = 1;
        double value = E_loc_avg(vl, wei);
        weights w = wei, w2 = wei;
        vector<function<visible_layer(const visible_layer, const weights &, std::mt19937_64 &)>> sampler_vector;
        for (size_t i = 0; i < 3; i++)
        {
            sampler_vector.push_back(sampler_mp);
        }

        vector<function<mat(const visible_layer, const weights &)>> matrix_maker;
        matrix_maker.push_back(vis_cross_tanh);
        matrix_maker.push_back(identity_vis_lay);
        matrix_maker.push_back(tanh_matrix);

        static gama g(gama_init_value, &n);

        vector<mat> W_update = inv_S_F(vl, w, sampler_vector, matrix_maker);

        (w.W) -= g * W_update[0]; // ((arma::norm(W_update[0])));
        (w.a) -= g * W_update[1]; // ((arma::norm(W_update[1])));
        (w.b) -= g * W_update[2]; // ((arma::norm(W_update[2])));
        return w;
    }

    double W_update(visible_layer &vl, weights &w)
    {
        weights temp_w;
        static int n = 1;
        temp_w = W_update_chooser(vl, w, n);
        w = temp_w;
        static gama g(gama_init_value, &n);
        n++;
        return g.out();
    }
}

#endif