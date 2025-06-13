#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

const int rows = 1;
const int columns = 5;
const int alpha = 4;

typedef arma::mat matrix;
namespace neural_net
{
	// just a thing to call visible layer
	struct visible_layer
	{
		matrix vis_lay = arma::randu(columns, 1);
	};
	// a class to simmulate the nodes(number is alpha) and the input 1d chain is column
	class Neural_net
	{
	private:
		// this is the matrix which is multiplied to the column vector for the node values to be obtained
		void Hidden_layer_init();

	public:
		double E_loc();
		visible_layer Vis_lay;
		double psi_s();
		matrix b = arma::zeros(alpha, 1);
		matrix hidden_layer;
		Neural_net();
	};
}
// implimentation
void neural_net::Neural_net::Hidden_layer_init()
{
	hidden_layer = arma::randu(alpha, columns);
	b = arma::randu(b.n_rows, b.n_cols);
	Vis_lay.vis_lay.for_each([](matrix::elem_type &m)
							 { (m > 0.5) ? (m = -1) : (m = 1); });
	psi_s();
}

double neural_net::Neural_net::psi_s()
{
	matrix m = hidden_layer * Vis_lay.vis_lay + b;
	double psi = 1;
	for (auto i : m)
	{
		psi = cosh(i) * psi;
	}
	return psi;
}
neural_net::Neural_net::Neural_net()
{
	Hidden_layer_init();
}

#endif