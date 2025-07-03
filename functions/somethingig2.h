#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
#include <matplot/matplot.h>

#include <vector>
#include <iostream>
#include <cmath>
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
    int row = 100;
    int alpha = 2;
    long double gama = 1;

    // parameters of the hamil
    long double h = 1;
    long double j = 1;
    std::random_device rd;

    // debuging
    int alpksp = 1;
    ;
    vector<double> int_n, somevar;

    const int dim = pow(2, row);
    struct weights
    {
        mat W;
        mat a;
        mat b;
        weights()
        {
            W = arma::randu(alpha, row);
            a = arma::randu(row, 1);
            b = arma::randu(alpha, 1);
        }
        void  operator/(double n)
        {
            W=W/n;
            a=a/n;
            b=b/n;
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

            S(i % row, 0) = -S(i % row, 0);
            // std::cout << "flip\n"
            //           << S(i, 0);
        }
    };

    mat theta_matrix(const visible_layer &VL, const weights &w)
    {
        mat M(alpha, 1);
        M = w.b + w.W * VL.S;
        return M;
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
        {
            {
                using namespace matplot;
                plot(int_n, somevar);
                save("./data/debug.png");
            }
            throw runtime_error("stfu");
        }
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
        for (size_t i = 0; i < alpha; i++)
        {
            cosh_theta = 2 * cosh_theta * cosh(m(i, 0));
        }
        // if (cosh_theta>1000)
        // {
        //     /* code */
        //     std::cout<<"\ncosh theta ="<<cosh_theta<<"\n";
        // }
        bruh(cosh_theta);
        bruh(sig_i_a_i);
        psi = cosh_theta * exp(sig_i_a_i);
        somevar.push_back(log(psi));
        int_n.push_back((alpksp));
        alpksp++;
        return log(psi);
    }
    visible_layer sampler(visible_layer VL, std::random_device &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist(0, VL.S.n_rows - 1);
        VL.flip(dist(rd));
        return VL;
    }

    long double p_ratio(visible_layer VL, visible_layer VL2, const weights &w)
    {
        bruh(psi(VL, w) / psi(VL2, w));
        double a = psi(VL, w), b = psi(VL2, w), c = a / b;

        return c;
    }

    long double E_loc(visible_layer VL, weights W)
    {

        // the hamiltonian is h*sum(sig_x)+ j*sum(sig_z(i)*sig_z(i+1))
        long double E_loc = 0;
        visible_layer m = VL;
        for (size_t i = 0; i < row; i++)
        {
            mat l = m.S;
            m.flip(i);
            E_loc += VL.S(i % row, 0) * VL.S((i + 1) % row) + h * p_ratio(m, VL, W);
            m.flip(i);
        }
        bruh(E_loc);
        if (std::isinf(E_loc))                       //! delete this
            throw runtime_error("shoul have known"); //! delete this;
        return E_loc;
    }

    long double E_loc_avg(const visible_layer VL, const weights &W, int itt_no = 100,
                          std::random_device &rd = pj::rd)
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
            VL2 = sampler(VL);
            e_loc += E_loc(VL2, W);
            n++;
        }

        // std::cout << "e_loc=" << e_loc << std::std::std::endl;
        // for (size_t i = 0; i < itt_no; i++) // itt for markov chain
        // {
        //     b = dist_int(rd);
        //     VL2 =sampler(VL);
        //     if (E_loc(VL2, W) < E_loc(VL3, W)) // if energyis less the sol is accepted
        //     {
        //         VL3 = VL2;
        //         e_loc += E_loc(VL2, W);
        //         // std::cout<<"E_loc=" <<E_loc(VL2, W)<<"\n";
        //         n++;
        //         continue;
        //     }
        //     else // if energy is more a coinn is tossed and is accepted if the prob is accepted
        //     {
        //         {
        //             e_loc += E_loc(VL2, W);
        //             // std::cout<<"E_loc=" <<E_loc(VL2, W)<<"\n";
        //             VL3 = VL2;
        //             n++;
        //             continue;
        //         }
        //     }
        // }
        bruh(e_loc / n);
        return e_loc / n;
    }
    void a_update(visible_layer vl, weights &w, int N = 1)
    {
        double lamda = pow(10, -2);
        // lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);
        mat O, OT_O, OT, O_O, E_OT,
            S, F, a_n, i = arma::eye(row, row);
        O = sampler(vl).S.t();
        OT = sampler(vl).S;
        O_O = sampler(vl).S;
        OT_O = (O_O) * (O_O.t());
        E_OT = E_loc(vl, w) * sampler(vl).S;
        for (size_t i = 1; i < N; i++)
        {
            O += sampler(vl).S.t();
            OT += sampler(vl).S;
            O_O = sampler(vl).S;
            OT_O += (O_O) * (O_O.t());
            E_OT += E_loc(vl, w) * sampler(vl).S;
        }
        long double e_loc = E_loc_avg(vl, w);

        S = (OT_O / N) - ((OT * O / pow(N, 2)));
        S = S - lamda * i;
        F = (E_OT / N) - e_loc * OT / N;
        // std::cout << "norm=" << arma::norm(gama * (arma::pinv(S)) * F) << endl;
        double update = arma::norm(gama * (arma::pinv(S)) * F);
        // if (std::isnan(arma::norm(gama * (arma::pinv(S)) * F)))
        // {
        //     {
        //         using namespace matplot;
        //         plot(int_n, somevar);
        //         hold(on);
        //         save("./data/debug.png");
        //     }
        //     throw runtime_error("stfu");
        // }
        w.a = w.a + gama * (arma::pinv(S)) * F/update;
    }
    void w_update(visible_layer vl, weights w)
    {
    
    }

}

#endif