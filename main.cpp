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
	if(n>a.size())
	return avg_cal(a);
	return reduce(a.end()-n , a.end()) / n;
}










int main()
{
	std::random_device rd;
	uniform_int_distribution<int> dist(0, pj::row - 1);
	auto start = std::chrono::high_resolution_clock::now();
	arma::arma_rng::set_seed_random();	
	number_of_sites = pj::row;
	J = pj::j;
	H=pj::h;
	hamiltoian_matrix matrix;
	arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	double min_value = matrix.min_eig_value();
	cout << "minimum eigen values are :\n"
		 << matrix.min_eig_value() << "\n";


	pj::visible_layer VL, VL2 = VL;
	pj::weights W;

	double alpha = pj::E_loc_avg(VL, W), beta = 0;
	int gama = 0, counting = 0;
	arma::mat a, b, w;
	vector<double> count, energy;

	VL = VL2;
	vector<double> e_loc, e_loc_avg, n;
	for (size_t j = 0; j < 10; j++)
	{
			for (size_t i = 0; i < 10000; i++)
			{
				pj::W_update(VL,W);

				e_loc.push_back(avg_cal(e_loc_avg,20));
				e_loc_avg.push_back(pj::E_loc_avg(VL, W));
				n.push_back(gama);
				gama++;
				
				if( gama % 50 ==0 )
				{
					// VL= pj::sampler(VL,W);
					cout<<"---------------------------------------------------------\n";
					double avg = avg_cal(e_loc_avg,50);
					cout<<"e loc avg per site is  ="<<avg/pj::row<<"\n"
					<<"exact value is \t\t="<<min_value/pj::row<<"\n"
					<<"and their difference is = "<<(avg-min_value)/pj::row<<"\n"
					<<"the percentage error is = "<<abs((avg-min_value)*100/(avg))<<"%\n"
					<<"";
					// cout<<"visible layer is ::"<<VL.S<<"\n";
					// cout<<pj::E_loc(VL,W)<<"is he value of e_loc\n";
					// plot(n,e_loc_avg, gama+2);
					// W.shake();
					// e_loc.clear();
					// e_loc_avg.clear();
					// n.clear();
				}
				VL=pj::sampler(VL,W);
			}
			plot(n, e_loc, 20+1);
			// W.shake();
		counting++;
		}

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
