#include <vector>
#include <time.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <armadillo>
#include <random>
#include <matplot/matplot.h>
#include "functions/ExactSol.h"
#include "functions/somethingig2.h"
using namespace std;

#define Number 10
void plot(vector<double> a, vector<double> b, vector<double> c, int i)
{
	using namespace matplot;
	plot(a, b, "3");
	hold(on);
	plot(a, c, "-o-");
	matplot::legend({"E loc", "E loc avg"});
	save("./data/image" + to_string(i) + ".png");
	cla();
}

int main()
{
	std::random_device rd;
	uniform_int_distribution<int> dist(0, pj::row - 1);
	auto start = std::chrono::high_resolution_clock::now();

	pj::row = 10;
	pj::visible_layer VL, VL2 = VL;
	pj::weights W;

	double alpha=pj::E_loc_avg(VL, W), beta=0;
	int gama=0;

	cout << alpha << endl;
	
	vector<double> e_loc, e_loc_avg, n;
	VL = VL2;
	for (size_t j = 0; j < 1000; j++)
	{
		/* code */
		
		for(size_t i = 0; i < 1000; i++)
		{
			cout<<j<<"\n";
			pj::a_update(VL, W);
			// std::cout <<alpha<< "\n";
		}
		n.push_back(gama);
		gama++;
		e_loc.push_back(alpha);
		VL.flip(dist(rd));
		alpha=pj::E_loc_avg(VL, W);
	}
	{
		using namespace matplot;
		plot(n,e_loc,"--");
		save("./debug.png");
		cin.get();
	}
	// plot(n,e_loc,e_loc_avg,i);

	// number_of_sites =10;
	// arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	// arma::mat vector=arma::zeros(hamiltonian.n_cols,1);
	// vector(5,0)=1;
	// // cout<<hamiltonian*vector;
	// cout<<"eigen values are :\n"<<matrix.Eigen_values<<"\n";
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
