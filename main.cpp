#include <vector>
#include <time.h>
#include <iostream>
#include <cmath>
#include<chrono>
#include<armadillo>
// #include "functions/ExactSol.h"
#include"functions/somethingig.h"
using namespace std; 

#define Lattice_type "l" //! lattice type l(linear), s(2d square), etc to be expanded not yet any fuctionality
#define Number 10


int main()
{
	
	auto start = std::chrono::high_resolution_clock::now();
	
	neural_net::Neural_net hello;
	cout<<hello.visible_layer()<<"\n";
	cout<<hello.S_H_state_vector()<<"\n";
	cout<<"e_loc = "<<hello.E_loc();

	
	
	// number_of_sites =3;
	// hamiltoian_matrix matrix;
	// arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	// arma::mat vector=arma::zeros(hamiltonian.n_cols,1);
	// vector(5,0)=1;
	// cout<<hamiltonian*vector;
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
// //constants
// sigma_n c;
// // matrix.printy();
// cout<<"eigen vectors are:\n "<<matrix.Eigen_states.col(get<0>(matrix.min_eig_value()))<<"\n";
// cout<<"eigen values are :\n"<<matrix.Eigen_values<<"\n";
// matrix.magnetization_calc((get<0>(matrix.min_eig_value())));
// vector<double> E_L,H;
// //auto a = arma::linspace(0,30,1000);
// // for (auto i :a)
// // {
	// // 	hamiltoian_matrix matrix(i);
	// // 	E_L.push_back(matrix.min_eig_value_per_site());
	// // 	H.push_back(i);
// // }
// // matplot::plot(H,E_L);//this function plots lowest energy eigen state vs h values
