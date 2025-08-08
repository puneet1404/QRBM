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
#include "functions/constants.h"
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
double avg_cal(const vector<double> &a, int n)
{

	// double count = static_cast<double>(a.size());
	if (n > a.size())
		return avg_cal(a);
	return reduce(a.end() - n, a.end()) / n;
}

int main()
{
	std::random_device rd;
	uniform_int_distribution<int> dist(0, pj::row - 1);
	auto start = std::chrono::high_resolution_clock::now();
	arma::arma_rng::set_seed_random();
	number_of_sites = pj::row;
	J = pj::J;
	H = pj::H;
	hamiltoian_matrix matrix;
	arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	double min_value = matrix.min_eig_value();
	cout << "minimum eigen values are :\n"
		 << matrix.min_eig_value() << "\n";

	pj::visible_layer VL, VL2 = VL;
	pj::weights W;

	int gama = 0;
	arma::mat a, b, w;
	vector<double> count, energy;

	VL = VL2;
	vector<double> e_loc, e_loc_avg, n;
	try
	{

		for (size_t j = 0; j < 100; j++)
		{
			for (size_t i = 0; i < 500; i++)
			{
				pj::W_update(VL, W);
				// cout<<i<<"\n";

				e_loc_avg.push_back((pj::E_loc_avg(VL, W)));
				e_loc.push_back((pj::E_loc(VL, W)));
				n.push_back(gama);
				gama++;
				if (gama % 10 == 0)
				{
					// VL= pj::sampler(VL,W);
					cout << "---------------------------------------------------------\n";
					double avg = avg_cal(e_loc_avg, 10);
					cout << "e loc avg per site is  =" << avg / pj::row << "\n"
						 << "e loc value per site is =" << avg_cal(e_loc, 20) / pj::row << "\n"
						 << "exact value is \t\t=" << min_value / pj::row << "\n"
						 << "and their difference is = " << (avg - min_value) / pj::row << "\n"
						 << "the percentage error is = " << abs((avg - min_value) * 100 / (avg)) << "%\n"
						 << "w is =\t\t\t" << arma::norm(W.W) << "\n"
						 << "a is =\t\t\t" << arma::norm(W.a) << "\n"
						 << "b is =\t\t\t" << arma::norm(W.b) << "\n";

					pj::gama g(1);
					plot(n, e_loc_avg, 20 + 1);
					plot(n, e_loc, 22);
					// W.shake(g);
				}
				// if( gama%100==0)
				// {

				// 	plot(n, e_loc, 20+1);
				// 	cout<<i<<"\n";
				// 	// VL=pj::sampler(VL,W);
				// }
			}
			if (pj::picture_rest)
			{
				n.clear();
				e_loc.clear();
				e_loc_avg.clear();
			}
			
		}
	}
	catch (runtime_error)
	{
		plot(n, e_loc, 20 + 1);
		cout << W.W << "\n";
		cout << W.b << "\n";
		cout << W.a << "\n";
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
