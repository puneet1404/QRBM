//alternate implimentaion of exact sol 1
#ifndef _ExactSol2_H_
#define _ExactSol2_H_

// ↑ ↓

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <complex>

int number_of_sites(4);
int dim(2);
std::complex<double> a(0,1);//just a stand in for comple no i;
const double h=1;//for hamiltonian equation 
const double J=2;//for hamiltonian equation 
using namespace std;

struct sigma_n//just for convinience; any changes for other spins should be updated here first 
{
    Eigen::Matrix2cd z{
        {1, 0},
        {0, -1}};
    Eigen::Matrix2cd x{
        {0, 1},
        {1, 0}};
    Eigen::Matrix2cd y{
        {0,-a},
        {a,0}
    };
    int n;
    char direc;
};

class hamiltoian_matrix//class to construct kroneker product and hamiltonian
{
private:
    vector<Eigen::MatrixXcd> X;
    vector<Eigen::MatrixXcd> Y;
    vector<Eigen::MatrixXcd> Z;
    vector<Eigen::MatrixXcd> hamil;
    const size_t num = number_of_sites;
    Eigen::MatrixXcd kron_prod_matrix(Eigen::MatrixXcd &initial_matrix, Eigen::MatrixXcd &placed_x);
    Eigen::MatrixXcd calc_hamiltonian();
    void kron_prod(sigma_n &n);
    
    public:
    hamiltoian_matrix();
    ~hamiltoian_matrix();
    Eigen::MatrixXcd Hamiltonian;
    void printx();
    void printy();
    void printz();
};

class state_vector//class for the state vector of the above hamiltonian, tho i think i should implement such a thing inside that class only.
{
private:
    std::vector<complex<double>> state;

public:
    state_vector(/* args */);
    ~state_vector();
};






/*

this is for implimentations

*/



state_vector::state_vector(/* args */)
{
}

state_vector::~state_vector()
{
}

hamiltoian_matrix::hamiltoian_matrix()
{
    sigma_n sigma;
    for (size_t i = 0; i < num; i++)
    {
        hamil.push_back(Eigen::MatrixXcd::Identity(dim, dim));
        X.push_back(Eigen::MatrixXcd::Identity(1, 1));
        Y.push_back(Eigen::MatrixXcd::Identity(1, 1));
        Z.push_back(Eigen::MatrixXcd::Identity(1, 1));
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
    Hamiltonian=calc_hamiltonian();
}
hamiltoian_matrix::~hamiltoian_matrix()
{
}

void hamiltoian_matrix::kron_prod(sigma_n &n)
{
    vector<Eigen::MatrixXcd> M;
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
        hamil[n.n] = Eigen::MatrixXcd::Identity(dim, dim);
    }
    else if ((n.direc == 'y') || (n.direc == 'Y'))
    {
        hamil[n.n] = n.y;
        for (size_t i = 0; i < num; i++)
        {
            Y[n.n] = kron_prod_matrix(Y[n.n], hamil[i]);
        }
        hamil[n.n] = Eigen::MatrixXcd::Identity(dim, dim);
    }
    else if ((n.direc == 'z') || (n.direc == 'Z'))
    {
        hamil[n.n] = n.z;
        for (size_t i = 0; i < num; i++)
        {
            Z[n.n] = kron_prod_matrix(Z[n.n], hamil[i]);
        }
        hamil[n.n] = Eigen::MatrixXcd::Identity(dim, dim);
    }
}
Eigen::MatrixXcd hamiltoian_matrix::kron_prod_matrix(Eigen::MatrixXcd &initial_matrix, Eigen::MatrixXcd &placed_sigma)
{
    int beta = initial_matrix.rows(), gama = initial_matrix.cols(), X_r = placed_sigma.rows(), X_c = placed_sigma.cols();
    Eigen::MatrixXcd alpha = Eigen::MatrixXcd::Zero(beta * X_r, gama * X_c);
    for (size_t j = 0; j < beta; j++) // for rows
    {
        for (size_t i = 0; i < gama; i++) // for cols
        {
            alpha.block(j * X_r, i * X_c, X_r, X_c) = (initial_matrix(j, i)) * placed_sigma;
        }
    }
    return alpha;
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