#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <unordered_map>

const int columns = 1;
const int rows = 5;
const int alpha = 4;
std::complex<double> coeff_sig_x = (1/2,0);
std::complex<double> coeff_sig_y = (0,-1/2);

typedef arma::mat matrix;
typedef arma::cx_mat cmatrix;
typedef std::complex<double> dcomplex;
namespace neural_net
{
	struct stocastic_visible_layer
	{
		int state_vector = 0;
		dcomplex co_efficient = (0, 0);
		stocastic_visible_layer(int stat_vec, dcomplex alpha = 0)
		{
			state_vector = stat_vec;
			co_efficient = 0;
		}
	};

	// just a thing to call visible layer
	struct visible_layer
	{
		matrix vis_lay = arma::randu(rows, 1);
		int state_vector = 0;
		void state_vector_init();
	};
	// a class to simmulate the nodes(number is alpha) and the input 1d chain is column
	class Neural_net
	{
	private:
		// this is the matrix which is multiplied to the column vector for the node values to be obtained
		void Hidden_layer_init();
		visible_layer Vis_lay;
		double decay_parameter = 0;
		void sigma(int location, char direc, matrix& config,std::vector<stocastic_visible_layer> map);
		
		public:
		matrix to_S(int n);
		int to_integer(matrix &S);
		double E_loc();
		matrix &visible_layer(); // done
		matrix b = arma::randu(alpha, 1);
		matrix a = arma::randu(rows, 1);
		matrix hidden_layer;
		double psi_s();			 // done
		double psi_s(matrix &S); // done
		double psi_s(int alpha);
		Neural_net();
	};
}

// implimentation

// constructor
neural_net::Neural_net::Neural_net()
{
	Hidden_layer_init();
	Vis_lay.state_vector_init();
}

// return visible layer as a object
matrix &neural_net::Neural_net::visible_layer()
{
	return Vis_lay.vis_lay;
}

// makes  the hidden layer and gives it random values;
void neural_net::Neural_net::Hidden_layer_init()
{
	hidden_layer = arma::randu(alpha, rows);
	b = arma::randu(b.n_rows, b.n_cols);
}

/// various implimnetations to calculate the probalility distribution;
double neural_net::Neural_net::psi_s()
{
	matrix m = hidden_layer * Vis_lay.vis_lay + b;
	double psi = 1;
	for (auto i : m)
	{
		psi = (cosh(i)) * psi;
	}
	return psi * exp(arma::as_scalar(a.t() * visible_layer()));
}

// gives psi_s for a sfecific value of the input layer;
double neural_net::Neural_net::psi_s(matrix &S)
{
	matrix m = hidden_layer * S + b;
	double psi = 1;
	for (auto i : m)
	{
		psi = (cosh(i)) * psi;
	}
	return psi * exp(arma::as_scalar(a.t() * S));
}

// gives psi_s for specific statevector in the form of an integer
double neural_net::Neural_net::psi_s(int alpha)
{
	matrix S = to_S(alpha);
	return psi_s(S);
}

// converts an integer into a visible layer vector
matrix neural_net::Neural_net::to_S(int alpha)
{
	matrix S = arma::zeros(rows, 1);
	int number = log2(alpha) + 1;
	for (size_t i = 1; i < rows + 1; i++)
	{
		(alpha % 2 == 1) ? (S[rows - i] = -1) : (S[rows - i] = 1);
		alpha = alpha >> 1;
	}
	return S;
}
//from visible layer to an integer map (unique for a given set of lattice of size (n))
int neural_net::Neural_net::to_integer(matrix &S)
{
	int n = 0;
	for (int i = 0; i < S.n_rows ; i++)
	{
		(S(i, 0) == 1) ? (n = n << 1) : (n = (n << 1) + 1);
	}
	return n;
}
void neural_net::Neural_net::sigma(int location, char direc, matrix& config,std::vector<stocastic_visible_layer> map)
{
	int alpha=1, beta= to_integer(config), place_holder;
	if ((direc == 'x') || (direc == 'X'))
	{
		alpha= (alpha<<(location%rows));
		place_holder = alpha^beta;
		stocastic_visible_layer m(place_holder, coeff_sig_x);
		map.push_back(m);
	}
	if ((direc == 'y') || (direc == 'Y'))
	{
		alpha= (alpha<<(location%rows));
		place_holder = alpha^beta;

		stocastic_visible_layer m(place_holder, coeff_sig_y);
		map.push_back(m);
	}
		if ((direc == 'Z') || (direc == 'z'))
		{
		alpha= (alpha<<(location%rows));
		place_holder = alpha^beta;

		stocastic_visible_layer m(place_holder, coeff_sig_y);
		map.push_back(m);
		}

}

// calculation of local energy
double neural_net::Neural_net::E_loc()
{
	std::vector<stocastic_visible_layer> map;
	for (size_t i = 0; i < rows; i++)
	{
		sigma(i,'x',visible_layer(),map);
		sigma(i,'z',visible_layer(),map);
	
}


//! implimnetation for visible_layer struct
void neural_net::visible_layer::state_vector_init()
{
	vis_lay.for_each([](matrix::elem_type &m)
					 { (m > 0.5) ? (m = -1) : (m = 1); });
	for (auto i : vis_lay)
	{
		(i == 1) ? ((state_vector = state_vector << 1)) : ((state_vector = (state_vector << 1) + 1));
	}
}
#endif