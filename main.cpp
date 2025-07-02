#include <vector>
#include <time.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <armadillo>
#include <random>
#include <matplot/matplot.h>
// #include "functions/ExactSol.h"
#include "functions/somethingig2.h"
using namespace std;

#define Number 10

int main()
{
	std::random_device rd;
	uniform_int_distribution<int> dist(0,pj::row-1);
	auto start = std::chrono::high_resolution_clock::now();

	vector<double> e_loc, e_loc_avg, n;
	pj::row = 10;
	pj::visible_layer VL;
	pj::weights W;
	cout<<W.W;
	cout << pj::E_loc(VL, W) << endl;
	for (size_t i = 0; i < 100; i++)
	{
		e_loc.push_back((pj::E_loc(VL,W)));
		e_loc_avg.push_back(pj::E_loc_avg(VL,W));
		n.push_back(i);
		VL.flip(dist(rd));	
	}
	cout << W.a;
	cout <<"e_loc="<< pj::E_loc(VL, W)<<endl;
	// e_loc.push_back(pj::E_loc(VL,W));
	// n.push_back(i);
	// e_loc_avg.push_back(pj::E_loc_avg(VL,W,100));
	
	{
		using namespace matplot;
		plot(n, e_loc,"-o");
		hold(on);
		plot(n,e_loc_avg,"--");
		matplot::legend({"E loc", "E loc avg"});
		cin.get();
	}
	// while (TRUE)
	{
		/* code */
	}
	
	// number_of_sites =10;
	// hamiltoian_matrix matrix;
	// arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	// arma::mat vector=arma::zeros(hamiltonian.n_cols,1);
	// vector(5,0)=1;
	// // cout<<hamiltonian*vector;
	// cout<<"eigen values are :\n"<<matrix.Eigen_values<<"\n";
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
