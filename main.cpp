#include <vector>
#include <time.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <armadillo>
#include <random>
#include <matplot/matplot.h>
#include <numeric>
#include <fstream>
#include <filesystem>
#include "functions/ExactSol.h"
#include "functions/RBM.h"
#include "functions/constants.h"
#include "functions/observales.h"
using namespace std;

#define Number 10
void plot(vector<double> a, vector<double> b, vector<double> c, double i)
{
	using namespace matplot;
	plot(a, b, "3");
	hold(on);
	plot(a, c, "-o-");
	matplot::legend({"E loc", "E loc avg"});
	save("./data/image" + to_string(i) + ".png");
	cla();
}
void plot(vector<double> a, vector<double> c, string name, double i)
{
	using namespace matplot;
	plot(a, c, "3");
	hold(on);
	matplot::legend({name});
	save("./data/image" + to_string(i) + ".png");
	cla();
}

void plot(vector<double> a, vector<double> b, double i)
{
	using namespace matplot;
	plot(a, b, "3");
	hold(on);
	save("./data/images_h_" + to_string(pj::H) + "__" + to_string(i) + ".png");
	cla();
}

void plot(vector<double> a, vector<double> b, double k, double i)
{
	vector<double> constant;
	for (auto i : a)
	{
		constant.push_back(k);
	}

	using namespace matplot;
	plot(a, b, "3");
	hold(on);
	plot(a, constant, "1");
	save("./data/images_h_" + to_string(pj::H) + "__" + to_string(i) + ".png");
	cla();
}

void save_vector(vector<double> vec, string name)
{
	filesystem::create_directories("./data_" + to_string(pj::row) + "_" + to_string(pj::H) + "_" + "c");
	ofstream file("./data_" + to_string(pj::row) + "_" + to_string(pj::H) + "_" + "c/" + name + ".txt");

	for (vector<double>::const_iterator i = vec.begin(); i != vec.end(); ++i)
	{
		file << *i << '\n';
	}
	file.close();
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
	static double H = pj::H;
	if (s)
	{
		if ((t != 0) && H == pj::H)
			return t;
		number_of_sites = pj::row;
		J = pj::J;
		H = pj::H;
		hamiltoian_matrix matrix;
		arma::cx_dmat hamiltonian = matrix.Hamiltonian;
		t = matrix.min_eig_value();
		std::cout << "\n"
				  << matrix.Eigen_values << "\n";
		H = pj::H;
		return t;
	}
	else
	{

		if ((t != 0) && H == pj::H)
			return t;
		number_of_sites = 10;
		J = pj::J;
		H = pj::H;
		hamiltoian_matrix matrix;
		arma::cx_dmat hamiltonian = matrix.Hamiltonian;
		t = matrix.min_eig_value();
		std::cout << "\n"
				  << matrix.Eigen_values << "\n";
		H = pj::H;
		return t;
	}
}
double &mag_calc(bool s = pj::exact_cal_bool)
{
	static double t = 0;
	static int n = 0;
	if (s)
	{
		if (n != 0)
			return t;
		number_of_sites = pj::row;
		J = pj::J;
		H = pj::H;
		hamiltoian_matrix matrix;
		arma::cx_dmat hamiltonian = matrix.Hamiltonian;
		matrix.min_eig_value();
		cout << "\n"
			 << matrix.magnetization_calc() << "\n";
		t = real(matrix.magnetization_calc()(0, 0));
		n++;
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
void print_info(pj::weights W, pj::magnetization mag, int gama, double g, vector<double> e_loc_avg,
				vector<double> e_loc, vector<double> magnetization, vector<double> mag_part,
				vector<double> n)

{
	cout << "---------------------------------------------------------\n";
	double avg = avg_cal(e_loc_avg, pj::run_avg_win);
	long double mag_avg = avg_cal(magnetization, pj::run_avg_win);

	cout << "e loc avg per site is  =" << avg << "\n"
		 << "e loc value per site is =" << avg_cal(e_loc, pj::run_avg_win) << "\n"
		 << ((pj::exact_cal_bool) ? ("exact value is \t\t=") : ("the previous value is\t="))
		 << min_eigen_value() / 10 //
								   // ((pj::exact_cal_bool) ? (pj::row) : (1))
		 << "\n"

		 << "and their difference is = " << (avg - min_eigen_value() / ((pj::exact_cal_bool) ? (pj::row) : (1))) << "\n"
		 << "the percentage error is = " << abs((avg - min_eigen_value() / ((pj::exact_cal_bool) ? (pj::row) : (1))) * 100 / (avg)) << "%\n"
		 << "magnetization in z direction (calc) = " << mag_avg << "\n"
		 << "magnetization in z direction (" << ((pj::exact_cal_bool) ? ("exact") : ("prev"))
		 << ")= " << mag_calc() << "\n"
		 << "error in magnetization is =" << 100 * (mag_calc() / pj::row - mag_avg) / (mag_calc() / pj::row) << "\n"

		 << "w is \t\t\t=" << arma::norm(W.W) << "\n"
		 << "a is \t\t\t=" << arma::norm(W.a) << "\n"
		 << "b is \t\t\t=" << arma::norm(W.b) << "\n"
		 << "gamma is \t\t=" << g << "\n"
		 << "beta is \t\t=" << pj::beta << "\n"
		 //  << "spin flip number \t="<<pj::spin_flip_num<<"\n"
		 << "this is the  " << gama << "th turn" << "\n"
		 //  <<"a=\n"<<W.a<<"\n"
		 //  <<"b=\n"<<W.b<<"\n"
		 //  <<"w=\n"<<W.W<<"\n"
		 << endl;
	plot(n, e_loc_avg, min_eigen_value() / 10, 1);
	plot(n, e_loc, min_eigen_value() / 10, 2);
	plot(n, magnetization, mag_calc() / 10, 3);
	plot(n, mag_part, mag_calc() / 10, 4);
	// if (!pj::exact_cal_bool)
	// {
	// 	mag_calc() = mag_avg;
	// 	min_eigen_value() = avg;
	// }
}

void run(double H_val)
{
	J = pj::J;
	H = H_val;
	pj::H = H_val;
	// cout << mag_calc();
	static pj::visible_layer VL;
	static pj::weights W;
	pj::magnetization mag(&VL, &W, pj::sampler_mp);
	double g = 0;
	double m = 0;

	// pj::beta = 1;

	static int gama = 0;
	static vector<double> e_loc, e_loc_avg, n, magnetization, mag_partition,
		mag_av, mag_pat_av;
	try
	{

		for (size_t j = 0; j < 20; j++)
		{
			for (size_t i = 0; i < 100; i++)
			{
				gama++;
				// std::cout << gama << "\n";
				g = pj::W_update(VL, W);
				// cout << "w is \t\t\t=" << arma::norm(W.W) << "\n"
				// 	 << "a is \t\t\t=" << arma::norm(W.a) << "\n"
				// 	 << "b is \t\t\t=" << arma::norm(W.b) << "\n";
				e_loc_avg.push_back((pj::E_loc_avg(VL, W))/ pj::row);
				e_loc.push_back(avg_cal(e_loc_avg, pj::run_avg_win));
				n.push_back(gama);
				magnetization.push_back(mag.mag_x_avg() / (pj::row));
				mag_partition.push_back(mag.mag_part_x_avg() / (pj::row * 0.2));
				mag_av.push_back(avg_cal(magnetization, pj::run_avg_win));
				mag_pat_av.push_back(avg_cal(mag_partition, pj::run_avg_win));

				if (gama % pj::plot_interval == 0 && pj::display_togle)
					print_info(W, mag, gama, g, e_loc_avg, e_loc, mag_av, mag_pat_av, n);
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
			save_vector(magnetization, "magnetization");
			save_vector(e_loc_avg, "e_loc_avg");
			save_vector(e_loc, "e_loc");
			save_vector(mag_partition, "mag_partition");
			if (pj::picture_rest)
			{
				n.clear();
				e_loc.clear();
				e_loc_avg.clear();
			}
			// for (size_t i = 0; i < pj::row; i++)
			// {
			// 	std::random_device rd;
			// 	uniform_int_distribution dist(0, pj::row - 1);
			// 	VL.flip(dist(rd));
			// }
		}
	}
	catch (runtime_error)
	{
		// plot(n, e_loc, 20 + 1);
		cout << W.W << "\n";
		cout << W.b << "\n";
		cout << W.a << "\n";
		cout << "eloc= " << pj::E_loc_avg(VL, W) << "\n";
	}
}

int main()
{
	std::random_device rd;
	// uniform_int_distribution<int> dist(0, pj::row - 1);
	auto start = std::chrono::high_resolution_clock::now();
	arma::arma_rng::set_seed_random();
	// cout<<"enter the value of H \n";
	// cin>>pj::H;
	// min_eigen_value();

	number_of_sites = pj::row;

	// vector<double> H_vec = {1.2};

	for (int i = 0; i < 1; i++)
	{
		run(.5);
		// pj::spin_flip_num++;
	}

	// J = pj::J;
	// H = pj::H;
	// cout << mag_calc();
	// pj::visible_layer VL;
	// pj::weights W;
	// pj::magnetization mag(&VL, &W, pj::sampler_mp);
	// double g = 0;
	// double m = 0;

	// int gama = 0;
	// vector<double> e_loc, e_loc_avg, n, magnetization, mag_partition,
	// 	mag_av, mag_pat_av;
	// try
	// {

	// 	for (size_t j = 0; j < 1000; j++)
	// 	{
	// 		for (size_t i = 0; i < 100; i++)
	// 		{
	// 			gama++;
	// 			std::cout << gama << "\n";
	// 			g = pj::W_update(VL, W);
	// 			// cout<<i<<"\n";

	// 			e_loc_avg.push_back((pj::E_loc_avg(VL, W)) / pj::row);
	// 			e_loc.push_back(avg_cal(e_loc_avg, pj::run_avg_win));
	// 			n.push_back(gama);
	// 			magnetization.push_back(mag.mag_x_avg() / (pj::row));
	// 			mag_partition.push_back(mag.mag_part_x_avg() / (pj::row * 0.2));
	// 			mag_av.push_back(avg_cal(magnetization, pj::run_avg_win));
	// 			mag_pat_av.push_back(avg_cal(mag_partition, pj::run_avg_win));

	// 			if (gama % pj::plot_interval == 0 && pj::display_togle)
	// 				print_info(W, mag, gama, g, e_loc_avg, e_loc, mag_av, mag_pat_av, n);
	// 			if (pj::graph_clear_after_interval && ((pj::graph_clear_interval == 0) ? (1) : (gama % (pj::graph_clear_interval)) == 0))
	// 			{
	// 				n.clear();
	// 				e_loc.clear();
	// 				e_loc_avg.clear();
	// 				magnetization.clear();
	// 			}

	// 			if (gama == pj::graph_cuttoff)
	// 			{
	// 				n.clear();
	// 				e_loc.clear();
	// 				e_loc_avg.clear();
	// 			}
	// 		}
	// 		save_vector(magnetization, "magnetization");
	// 		save_vector(e_loc_avg, "e_loc_avg");
	// 		save_vector(e_loc, "e_loc");
	// 		save_vector(mag_partition, "mag_partition");
	// 		if (pj::picture_rest)
	// 		{
	// 			n.clear();
	// 			e_loc.clear();
	// 			e_loc_avg.clear();
	// 		}
	// 	}
	// }
	// catch (runtime_error)
	// {
	// 	// plot(n, e_loc, 20 + 1);
	// 	cout << W.W << "\n";
	// 	cout << W.b << "\n";
	// 	cout << W.a << "\n";
	// 	cout << "eloc= " << pj::E_loc_avg(VL, W) << "\n";
	// }

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "\nTime taken by main function: " << std::chrono::duration_cast<std::chrono::minutes>(elapsed).count() << "mins\n";
}
