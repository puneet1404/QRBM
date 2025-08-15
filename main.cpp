#include <vector>
#include <time.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <armadillo>
#include <random>
#include <matplot/matplot.h>
#include "functions/ExactSol.h"
#include "functions/RBM.h"
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
void plot(vector<double> a, vector<double> c, string name, int i)
{
	using namespace matplot;
	plot(a, c, "3");
	hold(on);
	matplot::legend({name});
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

double &min_eigen_value(bool s = pj::exact_cal_bool)
{
	static double t = 0;
	if (s)
	{
		if (t != 0)
			return t;
		number_of_sites = pj::row;
		J = pj::J;
		H = pj::H;
		hamiltoian_matrix matrix;
		arma::cx_dmat hamiltonian = matrix.Hamiltonian;
		t = matrix.min_eig_value();
		return t;
	}
	else
	{
		return t;
	}
}
double &mag_calc(bool s = pj::exact_cal_bool)
{
	static double t = 0;
	if (s)
	{
		if (t != 0)
			return t;
		number_of_sites = pj::row;
		J = pj::J;
		H = pj::H;
		hamiltoian_matrix matrix;
		arma::cx_dmat hamiltonian = matrix.Hamiltonian;
		matrix.min_eig_value();
		cout<<"\n"<<matrix.magnetization_calc()<<"\n";
		t = real(matrix.magnetization_calc()(0, 0));
		return t;
	}
	else
	{
		return t;
	}
}
string cout_str(bool s = pj::exact_cal_bool)
{
	if (s)
	{
		return ("exact value is \t\t=");
	}
	return ("the previous value is\t=");
}
void print_info(pj::weights W, int gama, double g, vector<double> e_loc_avg, vector<double> e_loc, vector<double> mag, vector<double> n)

{
	cout << "---------------------------------------------------------\n";
	double avg = avg_cal(e_loc_avg, pj::run_avg_win);
	long double mag_avg = avg_cal(mag, 20);
	cout << "e loc avg per site is  =" << avg / pj::row << "\n"
		 << "e loc value per site is =" << avg_cal(e_loc, pj::run_avg_win) / pj::row << "\n"
		 << ((pj::exact_cal_bool) ? ("exact value is \t\t=") : ("the previous value is\t=")) << min_eigen_value() / pj::row << "\n"
		 << "and their difference is = " << (avg - min_eigen_value()) / pj::row << "\n"
		 << "the percentage error is = " << abs((avg - min_eigen_value()) * 100 / (avg)) << "%\n"
		 << "magnetization in x direction (calc) = " << mag_avg << "\n"
		 << "magnetization in x direction (exact)= " << mag_calc() << "\n"
		 << "error in magnetization is =" << (mag_calc() - mag_avg) << "\n"

		 << "w is \t\t\t=" << arma::norm(W.W) << "\n"
		 << "a is \t\t\t=" << arma::norm(W.a) << "\n"
		 << "b is \t\t\t=" << arma::norm(W.b) << "\n"
		 << "gamma is \t\t=" << g << "\n"
		 << "this is the  " << gama << "th turn" << endl;
	plot(n, e_loc_avg, 20 + 1);
	plot(n, e_loc, 22);
	plot(n, mag, "magnetizationexact value is "+ to_string(mag_calc()), 23);
	// W.shake(g);
	if (!pj::exact_cal_bool)
	{
		mag_calc()=mag_avg;
		min_eigen_value() = avg;
	}
}
int main()
{
	std::random_device rd;
	// uniform_int_distribution<int> dist(0, pj::row - 1);
	auto start = std::chrono::high_resolution_clock::now();
	arma::arma_rng::set_seed_random();

	// number_of_sites = pj::row;
	// J = pj::J;
	// H = pj::H;
	// hamiltoian_matrix matrix;
	// arma::cx_dmat hamiltonian = matrix.Hamiltonian;
	// double min_value = matrix.min_eig_value();
	// cout << "minimum eigen values are :\n"
	// 	 << matrix.min_eig_value() << "\n";

	pj::visible_layer VL;
	pj::weights W;

	double g = 0;
	double m = 0;

	int gama = 0;
	vector<double> e_loc, e_loc_avg, n, magnetization;
	try
	{

		for (size_t j = 0; j < 100; j++)
		{
			for (size_t i = 0; i < 500; i++)
			{
				gama++;
				g = pj::W_update(VL, W);
				// cout<<i<<"\n";

				e_loc_avg.push_back((pj::E_loc_avg(VL, W)));
				e_loc.push_back(avg_cal(e_loc_avg, pj::run_avg_win));
				n.push_back(gama);
				magnetization.push_back(pj::magnetization_avg_x(VL, W));

				if (gama % pj::plot_interval == 0 && pj::display_togle)
					print_info(W, gama, g, e_loc_avg, e_loc, magnetization, n);
				if (pj::graph_clear_after_interval && ((pj::graph_clear_interval == 0) ? (1) : (gama % (pj::graph_clear_interval)) == 0))
				{
					n.clear();
					e_loc.clear();
					e_loc_avg.clear();
					magnetization.clear();
				}

				if (gama == pj::graph_cuttoff)
				{
					n.clear();
					e_loc.clear();
					e_loc_avg.clear();
				}
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
		cout << "eloc= " << pj::E_loc_avg(VL, W) << "\n";
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "milliseconds\n";
}
