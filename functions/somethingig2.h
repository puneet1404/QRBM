#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
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
    int row = 10;
    int alpha = 2;
    long double gama = 1;

    // parameters of the hamil
    long double h = 1;
    long double j = 1;
    std::random_device rd;

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

    long double psi(visible_layer VL, const weights &WEI) // to calculate the probability psi(s) for a given weights and visible layers
    {
        long double psi = 0; // initialization of psi

        // sigmai* a(i) implimnetation
        long double sig_i_a_i = 0;
        sig_i_a_i = arma::as_scalar(VL.S.t()*WEI.a);

        // cosh theta implimentation
        long double cosh_theta = 1;
        mat m = theta_matrix(VL, WEI);
        for (size_t i = 0; i < alpha; i++)
        {
            cosh_theta = 2 * cosh_theta * cosh(m(i, 0));
        }

        psi = cosh_theta * exp(sig_i_a_i);
        return log(psi);
    }

    long double p_ratio(visible_layer VL, visible_layer VL2, const weights &w)
{
    return psi(VL, w) / psi(VL2, w);
    }

    long double E_loc(visible_layer VL, weights W)
    {

        // the hamiltonian is h*sum(sig_x)+ j*sum(sig_z(i)*sig_z(i+1))
        long double E_loc = 0;
        visible_layer m = VL;
        for (size_t i = 0; i < row-1; i++)
        {
            m.flip(i);
            E_loc += VL.S(i % row, 0) * VL.S((i + 1) /*% row*/) + h * p_ratio(m, VL, W);
            m.flip(i);
        }
        return E_loc;
    }

    long double E_loc_avg(const visible_layer VL, const weights &W, int itt_no=100,
                          std::random_device &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist_int(0, row - 1);
        std::uniform_real_distribution<long double> dist_real(0, 1);
        int n = 1;
        long double sum = 0,
                    a = 0,
                    e_loc = E_loc(VL, W);
        visible_layer VL2 = VL,
                      VL3 = VL;

        // std::cout << "e_loc=" << e_loc << std::std::std::endl;
        for (size_t i = 0; i < itt_no; i++) // itt for markov chain
        {
            a = dist_int(rd);
            VL2.flip(a);
            if (E_loc(VL2, W) < E_loc(VL3, W)) // if energyis less the sol is accepted
            {
                VL3 = VL2;
                e_loc += E_loc(VL2, W);
                // std::cout<<"E_loc=" <<E_loc(VL2, W)<<"\n";
                n++;
                continue;
            }
            else // if energy is more a coinn is tossed and is accepted if the prob is accepted
            {
                a = dist_real(rd);
                if (a < pow(p_ratio(VL2, VL3, W), 2))
                {
                    e_loc += E_loc(VL2, W);
                    // std::cout<<"E_loc=" <<E_loc(VL2, W)<<"\n";
                    VL3 = VL2;
                    n++;
                    continue;
                }
            }
        }
        return e_loc / n;
    }
    visible_layer sampler(visible_layer VL, std::random_device &rd = pj::rd)
    {
        std::uniform_int_distribution<int> dist(0, VL.S.n_rows - 1);
        VL.flip(dist(rd));
        return VL;
    }
    void a_update(visible_layer vl, weights &w, int N = 1)
    {
        static double lamda = 100, a = lamda;
        a = a * 0.9;
        lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);
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
        w.a = w.a - gama * (arma::pinv(S)) * F;
        // std::cout<<"w.a= "<<w.a<<"\n";
    }
    void w_update(visible_layer vl, weights &w, int N = 1)
    {
        static double lamda = 1, a = lamda;
        // a = a * 0.9;
        // lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);
        mat O, OT_O, OT, O_O, E_OT,
            S, F, a_n, i = arma::eye(row, row);
        O = theta_matrix(vl,w)*(vl.S.t());//alpha X 1 * 1 X row =alpha X row 
        OT = O.t();//row X alpha
        OT_O = (OT)*O; //row X alpha * alpha X row = row * row 
        E_OT = E_loc(vl, w) * OT; //row X alpha 
        for (size_t i = 1; i < N; i++)
        {
            auto a=sampler(vl);
        O = theta_matrix(a,w)*(a.S.t());//alpha X 1 * 1 X row =alpha X row 
        OT = O.t();//row X alpha
        OT_O = (OT)*O; //row X alpha * alpha X row = row * row 
        E_OT = E_loc(a, w) * OT; //row X alph
        }
        long double e_loc = E_loc_avg(vl, w);

        S = (OT_O / N) - ((OT * O / pow(N, 2)));//row X row
        S = S - lamda * i;//row X row 
        F = (E_OT / N) - e_loc * OT / N;//row x alpha 
        // std::cout << "update=" <<(gama * (arma::pinv(S)) * F) << "lmada=" << lamda;
          mat update = (gama * (arma::pinv(S)) * F);
         w.W = w.W - (gama * (arma::pinv(S)) * F).t();
        // std::cout<<w.W<<"\n";
    }
    void b_update(visible_layer vl, weights &w, int N = 1)
    {
        static double lamda = 100, a = lamda;
        a = a * 0.9;
        lamda = (a < pow(10, -4)) ? (pow(10, -4)) : (a);
        mat O, OT_O, OT, O_O, E_OT,
            S, F, a_n, i = arma::eye(row, row);
        O = theta_matrix(vl,w);//alpha X 1  =alpha X 1
        OT = O.t();//1 X alpha
        OT_O = (OT)*O; //1 X alpha * alpha X 1 = 1 * 1 
        E_OT = E_loc(vl, w) * OT; //1 X alpha 
        for (size_t i = 1; i < N; i++)
        {
            auto a=sampler(vl);
        O = theta_matrix(a,w)*(a.S.t());//alpha X 1 * 1 X row =alpha X row 
        OT = O.t();//row X alpha
        OT_O = (OT)*O; //row X alpha * alpha X row = row * row 
        E_OT = E_loc(a, w) * OT; //row X alph
        }
        long double e_loc = E_loc_avg(vl, w);

        S = (OT_O / N) - ((OT * O / pow(N, 2)));//row X row
        S = S - lamda * i;//row X row 
        F = (E_OT / N) - e_loc * OT / N;//row x alpha 
        // std::cout << "update=" <<(gama * (arma::pinv(S)) * F) << "lmada=" << lamda;
        mat update = (gama * (arma::pinv(S)) * F);
        w.W = w.W - (gama * (arma::pinv(S)) * F).t();
        // std::cout<<w.W<<"\n";
    }
}

#endif