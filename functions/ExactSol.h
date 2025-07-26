#ifndef _ExactSol_H_
#define _ExactSol_H_

// ↑ ↓

#include <eigen3/Eigen/Dense>
#include <vector>
#include <armadillo>
#include <iostream>
#include <complex>
#include <set>

int number_of_sites(10);
int dim(2);
std::complex<double> a(0, 1);    // just a stand in for comple no i;
const double magnetic_field = 1; // for hamiltonian equation
const double J = 1;
const double H = 0.5;             // for hamiltonian equation
using namespace std;

struct sigma_n // just for convinience; any changes for other spins should be updated here first
{
    const arma::cx_mat22 z{
        {static_cast<std::complex<double>>(1), 0},
        {0, -1}};
    const arma::cx_mat22 x{
        {0, static_cast<std::complex<double>>(1)},
        {1, 0}};
    const arma::cx_dmat22 y{
        {0, -a},
        {a, 0}};
    int n;
    char direc;
};

class hamiltoian_matrix // class to construct kroneker product and hamiltonian
{
private:
    vector<arma::cx_dmat> X;
    vector<arma::cx_dmat> Y;
    vector<arma::cx_dmat> Z;
    vector<arma::cx_dmat> hamil;
    size_t num = number_of_sites;
    double h = magnetic_field;
    arma::cx_dmat kron_prod_matrix(arma::cx_dmat &initial_matrix, arma::cx_dmat &placed_x);
    arma::cx_dmat calc_hamiltonian();
    void kron_prod(sigma_n &n);

public:
    hamiltoian_matrix();
    hamiltoian_matrix(double l, double m, int n);
    ~hamiltoian_matrix();
    arma::cx_dmat Eigen_states;
    arma::vec Eigen_values;
    arma::cx_dmat Hamiltonian;
    void printx();
    void printy();
    void printz();
    double min_eig_value_per_site();
    double min_eig_value();
    arma::cx_vec3 magnetization_calc(int i);
};

/*

this is for implimentations

*/

hamiltoian_matrix::hamiltoian_matrix()
{
    sigma_n sigma;
    for (size_t i = 0; i < num; i++)
    {
        hamil.push_back(arma::eye<arma::cx_dmat>(dim, dim));
        X.push_back(arma::eye<arma::cx_dmat>(1, 1));
        Y.push_back(arma::eye<arma::cx_dmat>(1, 1));
        Z.push_back(arma::eye<arma::cx_dmat>(1, 1));
    }

    for (int i = 0; i < num; i++)
    {
        sigma.n = i;
        sigma.direc = 'x';
        kron_prod(sigma);
        sigma.direc = 'z';
        kron_prod(sigma);
        sigma.direc = 'y';
        kron_prod(sigma);
    }
    Hamiltonian = calc_hamiltonian();
    arma::eig_sym(Eigen_values, Eigen_states, Hamiltonian);
}

hamiltoian_matrix::hamiltoian_matrix(double l /*value of h for magnetic field */,
                                     double m = 1 /* value of the coupling constant*/, int n = number_of_sites /*lattice size*/)
{
    num = n;
    h = l;
    sigma_n sigma;
    for (size_t i = 0; i < num; i++)
    {
        hamil.push_back(arma::eye<arma::cx_dmat>(dim, dim));
        X.push_back(arma::eye<arma::cx_dmat>(1, 1));
        Y.push_back(arma::eye<arma::cx_dmat>(1, 1));
        Z.push_back(arma::eye<arma::cx_dmat>(1, 1));
    }

    for (int i = 0; i < num; i++)
    {
        sigma.n = i;
        sigma.direc = 'x';
        kron_prod(sigma);
        sigma.direc = 'z';
        kron_prod(sigma);
        sigma.direc = 'y';
        kron_prod(sigma);
    }
    Hamiltonian = calc_hamiltonian();
    arma::eig_sym(Eigen_values, Eigen_states, Hamiltonian);
}

hamiltoian_matrix::~hamiltoian_matrix()
{
}

void hamiltoian_matrix::kron_prod(sigma_n &n)
{
    vector<arma::cx_dmat> M;
    if ((n.direc == 'x') || (n.direc == 'X'))
    {
        static int NO = 0;
        NO++;
        // cout<<"x number_of_sites ="<<NO<<"\n";
        hamil[n.n] = n.x;
        for (size_t i = 0; i < num; i++)
        {
            X[n.n] = kron_prod_matrix(X[n.n], hamil[i]);
        }
        hamil[n.n] = arma::eye<arma::cx_dmat>(dim, dim);
    }
    else if ((n.direc == 'y') || (n.direc == 'Y'))
    {
        hamil[n.n] = n.y;
        for (size_t i = 0; i < num; i++)
        {
            Y[n.n] = kron_prod_matrix(Y[n.n], hamil[i]);
        }
        hamil[n.n] = arma::eye<arma::cx_dmat>(dim, dim);
    }
    else if ((n.direc == 'z') || (n.direc == 'Z'))
    {
        hamil[n.n] = n.z;
        for (size_t i = 0; i < num; i++)
        {
            Z[n.n] = kron_prod_matrix(Z[n.n], hamil[i]);
        }
        hamil[n.n] = arma::eye<arma::cx_mat>(dim, dim);
    }
}
arma::cx_dmat hamiltoian_matrix::kron_prod_matrix(arma::cx_dmat &initial_matrix, arma::cx_dmat &placed_sigma)
{
    int beta = initial_matrix.n_rows, gama = initial_matrix.n_cols, X_r = placed_sigma.n_rows, X_c = placed_sigma.n_cols;
    arma::cx_dmat alpha(beta * X_r, gama * X_c);
    for (size_t j = 0; j < beta; j++) // for rows
    {
        for (size_t i = 0; i < gama; i++) // for cols
        {
            alpha(j * X_r, i * X_c, arma::size(placed_sigma)) = (initial_matrix(j, i)) * placed_sigma;
        }
    }
    return alpha;
}

double hamiltoian_matrix::min_eig_value_per_site()
{
    double lowest_value = 0;
    for (auto i : Eigen_values)
    {
        if (i < lowest_value)
            lowest_value = i;
    }
    return lowest_value / num;
}

double hamiltoian_matrix::min_eig_value()
{
    double lowest_value = Eigen_values[0];
    int n = 0;
    for (int i = 1; i < num; i++)
    {
        if (Eigen_values[i] < lowest_value)
        {
            n = i;
            lowest_value = Eigen_values[i];
        }
    }
    return lowest_value;
}

// arma::cx_vec3 hamiltoian_matrix::magnetization_calc(int n)
// {
//     arma::cx_dmat X_sum, Y_sum, Z_sum;
//     double X_mag=0, Y_mag=0, Z_mag=0;
//     for (size_t i = 0; i < num; i++)
//     {
//         X_sum += X[i];
//         Y_sum += Y[i];
//         Z_sum += Z[i];
//     }
//     (cout<<(arma::trans(Eigen_states.col(n)) * X_sum * Eigen_states.col(n)));
//     cout<<((arma::trans(Eigen_states.col(n)) * Y_sum * Eigen_states.col(n)));
//     cout<<((arma::trans(Eigen_states.col(n)) * Z_sum * Eigen_states.col(n)));
//     // arma::cx_vec3 vec{X_mag, Y_mag, Z_mag};
//     //return vec;
// }

arma::cx_dmat hamiltoian_matrix::calc_hamiltonian()
{
	arma::cx_dmat hamiltonian = arma::zeros<arma::cx_dmat>(pow(dim,num),pow(dim,num));
	for (size_t i = 0; i < num-1; i++)
	{
		hamiltonian += -J*Z[i]*Z[(i+1)%num] - H* X[i];
	}
    hamiltonian+= -H*X[num-1];
	return hamiltonian;
}

void hamiltoian_matrix::printx()
{
    int n = 0;
    for (auto i : X)
    {
        n++;
        cout << "the sigmaX_" << n << " matrix is :\n";
        cout << i << "\n";
    }
}
void hamiltoian_matrix::printy()
{
    int n = 0;
    for (auto i : Y)
    {
        n++;
        cout << "the sigmaY_" << n << " matrix is :\n";
        cout << i << "\n";
    }
}
void hamiltoian_matrix::printz()
{
    int n = 0;
    for (auto i : Z)
    {
        n++;
        cout << "the sigmaZ_" << n << " matrix is :\n";
        cout << i << "\n";
    }
}

#endif