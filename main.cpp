#include <vector>
#include <time.h>
#include <iostream>
#include <cmath>
#include<chrono>
#include<armadillo>
#include<random>
#include "functions/ExactSol.h"
#include"functions/somethingig.h"
using namespace std; 

#define Lattice_type "l" //! lattice type l(linear), s(2d square), etc to be expanded not yet any fuctionality
#define Number 10


int main()
{
	
	auto start = std::chrono::high_resolution_clock::now();
	std::srand(time(nullptr));
	neural_net::Neural_net hello;
	cout<<"e_loc = "<<hello.E_loc()<<"\n\n";

	
	
	number_of_sites =10;
	hamiltoian_matrix matrix;
	arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	arma::mat vector=arma::zeros(hamiltonian.n_cols,1);
	vector(5,0)=1;
	// cout<<hamiltonian*vector;
	cout<<"eigen values are :\n"<<matrix.Eigen_values<<"\n";
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
