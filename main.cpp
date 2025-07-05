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
void plot(vector<double> a, vector<double> b, int i)
{
	using namespace matplot;
	plot(a, b, "3");
	hold(on);
	save("./data/images" + to_string(i) + ".png");
	cla();
}

double avg_cal( const vector<double> & a)
{
	double count = static_cast<double> ( a.size());
	return reduce(a.begin(),a.end())/count;

}
int main()
{
	std::random_device rd;
	uniform_int_distribution<int> dist(0, pj::row - 1);
	auto start = std::chrono::high_resolution_clock::now();
	
	number_of_sites =10;
	hamiltoian_matrix matrix;
	arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	// arma::mat vector= arma::zeros(hamiltonian.n_cols,1);
	// vector(5,0)=1;
	// cout<<hamiltonian*vector;
	cout<<"eigen values are :\n"<<matrix.min_eig_value()<<"\n";
	




	
	pj::row = 10;
	pj::visible_layer VL, VL2 = VL;
	pj::weights W;

	double alpha=pj::E_loc_avg(VL, W), beta=0;
	int gama=0,counting=0;
	arma::mat a ,b, w;
	cout << alpha << endl;
	vector<double> count , energy;
	
	VL = VL2;
	for (size_t j = 0; j < 10; j++)
	{
		vector<double> e_loc, e_loc_avg, n;
		for(size_t i = 0; i < pow(10,2); i++)
		{
			a= pj::a_update(VL, W);
			w=pj::w_update(VL,W);
			b=pj::b_update(VL,W);
			W.a=a;
			W.b=b;
			W.W=w;
			
			// cout<<"w=\n"<<W.W<<"\n";
			// cout<<"b=\n"<<W.b<<"\n";
			e_loc.push_back(pj::E_loc(VL,W));
			e_loc_avg.push_back(pj::E_loc_avg(VL,W));
			n.push_back(gama);
			gama++;
			// cout<<gama<<"\n";
		}
		for (size_t i = 0; i < 10; i++)
		{
			VL.flip(dist(rd));
		}
		counting++;
		energy.push_back(avg_cal(e_loc_avg));
		count.push_back(counting);


		plot(n,e_loc,e_loc_avg,j);
	}
	plot(count, energy, 999);

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
