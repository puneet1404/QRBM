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
    typedef arma::mat mat;
    typedef std::complex<long double> dclx;

    // defines the paramemeter of the sim
    int col = 1;
    int row = 10;
    int alpha = 2;
    int hid_node_num = alpha * row;
    long double gama()
    {
        static double gama = 1;
        // gama=0.99*gama;
        // gama=(gama<pow(10,-2))?(pow(10,-2)):(gama);
        return gama;
    }

    // parameters of the hamil
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
            W = arma::randu(hid_node_num, row);
            a = arma::randu(row, 1);
            b = arma::randu(hid_node_num, 1);
        }
        void operator/(double n)
        {
            W = W / n;
            a = a / n;
            b = b / n;
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
            // std::cout << "flip\n"
            //           << S(i, 0);
        }
    };

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
        return b;
    }

    //! delete this
    double sig_i_a_i(visible_layer VL, const weights &WEI)
    {

        long double sig_i_a_i = 0;
        for (size_t i = 0; i < row; i++)
        {
            sig_i_a_i += WEI.a(i, 0) * VL.S(i, 0);
        }
        return sig_i_a_i;
    }
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

        // cosh theta implimentation
        long double cosh_theta = 1;
        mat m = theta_matrix(VL, WEI);
        m.for_each([](auto &m)
                   { m = cosh(m); });
        for (size_t i = 0; i < hid_node_num; i++)
        {
            cosh_theta = 2 * cosh_theta * (m(i, 0));
        }
        // if (cosh_theta>1000)
        // {
        //     /* code */
        //     std::cout<<"\ncosh theta ="<<cosh_theta<<"\n";
        // }
        bruh(cosh_theta);
        bruh(sig_i_a_i);
        psi = log(cosh_theta) + (sig_i_a_i);
        return (psi);
    }
    visible_layer sampler(visible_layer VL, std::random_device &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist(0, VL.S.n_rows - 1);
        VL.flip(dist(rd));
        return VL;
    }

    long double p_ratio(visible_layer VL, visible_layer VL2, const weights &w)
    {
        double a = psi(VL, w), b = psi(VL2, w), c = a / b;
        bruh(psi(VL, w) / psi(VL2, w));

        return c;
    }

    long double E_loc(visible_layer VL, const weights &W)
    {

        // the hamiltonian is h*sum(sig_x)+ j*sum(sig_z(i)*sig_z(i+1))
        long double E_loc = 0;
        visible_layer m = VL;
        for (size_t i = 0; i < row; i++)
        {
            m.flip(i);
            E_loc += j * VL.S(i % row, 0) * VL.S((i + 1) % row) + h * p_ratio(m, VL, W);
            m.flip(i);
        }
        bruh(E_loc);
        return E_loc;
    }

    long double E_loc_avg(const visible_layer VL, const weights &W, int itt_no = 1000, std::random_device &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist_int(0, row - 1);
        std::uniform_real_distribution<long double> dist_real(0, 1);
        int n = 1, b;
        long double sum = 0,
                    a = 0,
                    e_loc = E_loc(VL, W);
        visible_layer VL2 = VL,
                      VL3 = VL;
        for (size_t i = 0; i < itt_no; i++)
        {
            VL2 = sampler(VL2);
            e_loc += E_loc(VL2, W);
            n++;
        }
        bruh(e_loc / n);
        return e_loc / n;
    }
    mat a_update(visible_layer vl, const weights &w, int N = 1000)
    {
        bruh(arma::norm(w.a));
        bruh(arma::norm(w.b));
        bruh(arma::norm(w.W));

        double lamda = pow(10, -2);
        // lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);
        mat O, OT_O, OT, O_O, E_OT,
            S, F, a_n, i = arma::eye(row, row);

        visible_layer vl2 = vl;
        O = sampler(vl).S.t();               // 1 X row
        OT = sampler(vl).S;                  // row X 1
        O_O = sampler(vl).S;                 // row X 1
        OT_O = (O_O) * (O_O.t());            // row X row
        E_OT = E_loc(vl, w) * sampler(vl).S; // row X 1
// #pragma omp parallel for num_threads(6)
        for (size_t j = 1; j < N; j++)
        {
            vl2 = sampler(vl2);
            O += sampler(vl2).S.t();
            OT += sampler(vl2).S;
            O_O = sampler(vl2).S;
            OT_O += (O_O) * (O_O.t());
            E_OT += E_loc(vl2, w) * sampler(vl2).S;
        }
        long double e_loc = E_loc_avg(vl2, w);

        S = (OT_O / N) - ((OT * O / pow(N, 2)));
        S = S - lamda * arma::diagmat(S);
        F = (E_OT / N) - e_loc * OT / N;
        double update = arma::norm((arma::pinv(S)) * F);
        // std::cout << "a\n"
        //           << arma::pinv(S) * F << "\n";

        // if (update > pow(10, -3))
        mat m = w.a - gama() * (arma::pinv(S)) * F / update;
        return m;
    }
    mat w_update(visible_layer vl, const weights &w, int N = 1000)
    {

        double lamda = 10;
        mat O, OT_O, OT, O_O, E_OT,
            S, F, i = arma::eye(row, row),
                  theta = theta_matrix(vl, w);

        visible_layer rough = sampler(vl);
        mat tanh_theta = tanh_matrix(vl, w);

        O = (rough.S * (tanh_theta.t())).t(); // row X 1 * 1 X hid_node_num = row X hid_node_num.t() = hid_node_num X row r
        OT = (rough.S * (tanh_theta.t()));
        // row X hid_node_num
        OT_O = ((rough.S * (tanh_theta.t()))) *
               ((rough.S * (tanh_theta.t())).t()); // row X row
        E_OT = E_loc(vl, w) *
               (rough.S * (tanh_theta.t())); // hid_node_num X row
// #pragma omp parallel for num_threads(6)
        for (size_t j = 1; j < N; j++)
        {
            rough = sampler(vl);
            tanh_theta = tanh_matrix(rough, w);
            O += (rough.S * (tanh_theta.t())).t();
            OT += (rough.S * (tanh_theta.t()));
            OT_O += (rough.S * (tanh_theta.t())) * (rough.S * (tanh_theta.t())).t(); // row X row
            E_OT += E_loc(vl, w) * (rough.S * (tanh_theta.t()));
        }
        long double e_loc = E_loc_avg(vl, w);

        S = (OT_O / N) - ((OT * O / pow(N, 2))); //
        S = S - lamda * i;
        F = (E_OT / N) - e_loc * OT / N;
        double update = arma::norm((arma::pinv(S)) * F);
        // std::cout << "w\n"
        //           << arma::pinv(S) * F << "\n";
        // if (update > pow(10, -3))
        mat m = w.W - gama() * ((arma::pinv(S)) * F).t() / update;
        return m;
    }
    mat b_update(visible_layer vl, const weights &w, int N = 1000)
    {
        static double lamda = pow(10, -2), a = 100;
        a = a * 0.9;
        lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);
        mat O, OT_O, OT, O_O, E_OT,
            S, F, i = arma::eye(hid_node_num, hid_node_num),
                  theta = theta_matrix(vl, w);
        visible_layer rough = (vl);
        mat tanh_theta = tanh_matrix(rough, w);

        O = tanh_theta.t();               // 1 X hid_node_num
        OT = tanh_theta;                  // hid_node_num X 1
        O_O = tanh_theta;                 // hid_node_num X 1
        OT_O = (O_O) * (O_O.t());         // hid_node_num X hid_node_num
        E_OT = E_loc(vl, w) * tanh_theta; // hid_node_num X 1
// #pragma omp parallel for num_threads(6)

        for (size_t j = 1; j < N; j++)
        {
            rough = sampler(vl);
            tanh_theta = tanh_matrix(rough, w);
            O += tanh_theta.t();
            OT += tanh_theta;
            O_O = tanh_theta;
            OT_O += (O_O) * (O_O.t());
            E_OT += E_loc(vl, w) * tanh_theta;
        }
        long double e_loc = E_loc_avg(vl, w);

        S = (OT_O / N) - ((OT * O / (N)*N)); // hid_node_num X hid_node_num
        S = S - lamda * arma::diagmat(S);
        F = (E_OT / N) - e_loc * OT / N;
        double update = arma::norm((arma::pinv(S)) * F);
        // std::cout << "b\n"
        //   << arma::pinv(S) * F << "\n";
        mat m = w.b - gama() * ((arma::pinv(S)) * F) / update;
        return m;
    }

}

#endif