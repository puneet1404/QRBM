#include <matplot/matplot.h>
#include <vector>
#include <time.h>
#include <iostream>
#include"functions/somethingig.h"
// #include "functions/ExactSol.h"
using namespace std;

#define Lattice_type "l" //! lattice type l(linear), s(2d square), etc to be expanded not yet any fuctionality
#define Number 10


int main()
{
	
	auto start = std::chrono::high_resolution_clock::now();
	
	neural_net::Neural_net hello;
	
	auto end = std::chrono::high_resolution_clock::now();
	cout<<"W=\n"<<hello.hidden_layer<<"\n"<<"b=\n"<<hello.b<<"\n"<<"s=\n"<<hello.visible_layer()<<
	"\n"<<"value of psi(s)="<<hello.psi_s(2)<<"\n"
	<<"the value of int(s) is "<<hello.to_integer(hello.visible_layer())<<endl;
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
// //constants
// number_of_sites =5;
// hamiltoian_matrix matrix;
// sigma_n c;
// arma::cx_dmat hamiltonian = matrix.Hamiltonian;
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
