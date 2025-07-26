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

double avg_cal(const vector<double> &a)
{
	double count = static_cast<double>(a.size());
	return reduce(a.begin(), a.end()) / count;
}
double avg_cal(const vector<double> &a, int n )
{
	// double count = static_cast<double>(a.size());
	return reduce(a.end()-n , a.end()) / n;
}
int main()
{
	std::random_device rd;
	uniform_int_distribution<int> dist(0, pj::row - 1);
	auto start = std::chrono::high_resolution_clock::now();

	number_of_sites = pj::row;
	hamiltoian_matrix matrix;
	arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	// arma::mat vector= arma::zeros(hamiltonian.n_cols,1);
	// vector(5,0)=1;
	// cout<<hamiltonian*vector;
	double min_value = matrix.min_eig_value();
	cout << "eigen values are :\n"
		 << matrix.min_eig_value() << "\n";



	pj::visible_layer VL, VL2 = VL;
	pj::weights W;

	double alpha = pj::E_loc_avg(VL, W), beta = 0;
	int gama = 0, counting = 0;
	arma::mat a, b, w;
	vector<double> count, energy;

	VL = VL2;
	for (size_t j = 0; j < 10; j++)
	{
			vector<double> e_loc, e_loc_avg, n;
			for (size_t i = 0; i < 1000; i++)
			{
				w = pj::w_update(VL, W);
				a = pj::a_update(VL, W);
				b = pj::b_update(VL, W);
				W.a = a;
				W.b = b;
				W.W = w;
				
				
				
				e_loc.push_back(pj::E_loc(VL, W));
				e_loc_avg.push_back(pj::E_loc_avg(VL, W));
				n.push_back(gama);
				gama++;
				if( gama % 100 ==0 )
				{
					cout<<"e loc avg is  ="<<avg_cal(e_loc_avg,100)<<"\n"
					<<"gama="<<pj::gama()<<"\n";
					plot(n, e_loc, gama+1);
					plot(n,e_loc_avg, gama+2);
					VL = pj::sampler(VL,W);

				}
				if( gama % 500 ==0 )
				{
					cout<<"i = "<<i<<"\t j= "<<j<<"\n";
					plot(n, e_loc,e_loc_avg, gama);
					VL = pj::sampler(VL,W);
					cout<<"e loc avg is  ="<<avg_cal(e_loc_avg,100)<<"\n"
					<<"gama="<<pj::gama()<<"\n";
					// e_loc_avg.clear();
					// n.clear();
				}
			}
			plot(n, e_loc,j);
		
		counting++;
		// energy.push_back(avg_cal(e_loc_avg));
		// count.push_back(counting);
	}
	// plot(count, energy, 999);
	// cout << "the vergaee energy diff is = " << abs(abs(min_value) - abs(avg_cal(energy)))<<"\n";
	// cout<<avg_cal(energy);

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
