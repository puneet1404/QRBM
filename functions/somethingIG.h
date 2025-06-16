#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <unordered_map>

const int columns = 1;
const int rows = 3;
const int alpha = 4;
std::complex<double> coeff_sig_x = std::complex<double>(1, 0);
std::complex<double> coeff_sig_y = std::complex<double>(0, -1);

typedef arma::mat matrix;
typedef arma::cx_mat cmatrix;
typedef std::complex<double> dcomplex;
namespace neural_net
{
	
	struct some_visible_layer
	{
		int state_vector = 0;
		dcomplex co_efficient = (0, 0);
		some_visible_layer(int stat_vec, dcomplex alpha = 0)
		{
			state_vector = stat_vec;
			co_efficient = alpha;
		}
		some_visible_layer operator+(some_visible_layer  alpha);
		some_visible_layer operator*(some_visible_layer  alpha);
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
		void sigma(int location, char direc, const matrix &config, std::vector<some_visible_layer> map);
		matrix to_S(int n);
		int to_integer(const matrix &S);
		cmatrix to_S(some_visible_layer alpha);
		cmatrix to_state(some_visible_layer alpha);
		some_visible_layer sigma(int location, char direc,const  matrix &config);

	public:
		matrix &visible_layer(); // done
		matrix b = arma::randu(alpha, 1);
		matrix a = arma::randu(rows, 1);
		matrix hidden_layer;
		cmatrix S_H_state_vector();
		double psi_s();			 // done
		double psi_s(matrix &S); // done
		double psi_s(int alpha);
		double E_loc();
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
	std::cout << "hehe2";
	std::cout << hidden_layer * S + b;
	matrix m = hidden_layer * S + b;
	std::cout << "hehe3";
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

cmatrix neural_net::Neural_net::to_S(some_visible_layer alpha)
{
	matrix m = to_S(alpha.state_vector);
	cmatrix beta = alpha.co_efficient * m;
	return beta;
}

// from visible layer to an integer map (unique for a given set of lattice of size (n))
int neural_net::Neural_net::to_integer(const matrix &S)
{
	int n = 0;
	for (int i = 0; i < S.n_rows; i++)
	{
		(S(i, 0) == 1) ? (n = n << 1) : (n = (n << 1) + 1);
	}
	return n;
}
void neural_net::Neural_net::sigma(int location, char direc,const matrix &config, std::vector<some_visible_layer> map)
{
	int alpha = 1, beta = to_integer(config), place_holder, just_sign = 0;
	if ((direc == 'x') || (direc == 'X'))
	{
		alpha = (alpha << (location % rows));
		place_holder = alpha ^ beta;
		some_visible_layer m(place_holder, coeff_sig_x);
		map.push_back(m);
	}
	if ((direc == 'y') || (direc == 'Y'))
	{
		alpha = (alpha << (location % rows));
		place_holder = alpha ^ beta;
		(beta < place_holder) ? (just_sign = 1) : (just_sign = -1);

		some_visible_layer m(place_holder, static_cast<double>(just_sign) * coeff_sig_y);
		map.push_back(m);
	}
	else
	{
		some_visible_layer m(to_integer(config), config(location, 0));
		map.push_back(m);
	}
}

neural_net::some_visible_layer neural_net::Neural_net::sigma(int location, char direc, const matrix &config)
{
	int alpha = 1, beta = to_integer(config), place_holder, just_sign = 0;
	if ((direc == 'x') || (direc == 'X'))
	{
		alpha = (alpha << (location % rows));
		place_holder = alpha ^ beta;
		some_visible_layer m(place_holder, coeff_sig_x);
		// std::cout<<(coeff_sig_x)<<"scalar value x\n";
		return m;
	}
	if ((direc == 'y') || (direc == 'Y'))
	{
		alpha = (alpha << (location % rows));
		place_holder = alpha ^ beta;
		(beta < place_holder) ? (just_sign = 1) : (just_sign = -1);

		some_visible_layer m(place_holder, static_cast<double>(just_sign) * coeff_sig_y);
		// std::cout<<static_cast<double>(just_sign) * coeff_sig_y<<"scalar value y\n";
		return m;
	}
	else
	{
		// std::cout<<arma::as_scalar(config(location%rows, 0))<<"scalar value z\n";
		some_visible_layer m(to_integer(config), arma::as_scalar(config(location%rows, 0)));
		return m;
	}
}
cmatrix neural_net::Neural_net::to_state(some_visible_layer alpha)
{
	cmatrix M;
	M(alpha.state_vector, 0) = 1;
	M = alpha.co_efficient * M;
	return M;
}
//! unfinished
// calculation of local energy
double neural_net::Neural_net::E_loc()
{
	std::vector<some_visible_layer> map;
	cmatrix bra_s_H = S_H_state_vector();
	
}

// gives a state vector
cmatrix neural_net::Neural_net::S_H_state_vector()
{
	
	cmatrix M;
	M.zeros(pow(2,rows),1); 
	std::vector<some_visible_layer> alpha;
	some_visible_layer beta(to_integer(Vis_lay.vis_lay),0);
	for (size_t i = 0; i < rows; i++)
	{
		alpha.push_back(sigma(i,'x',Vis_lay.vis_lay));
		beta = beta + (sigma(i, 'z',Vis_lay.vis_lay))*(sigma((i+1),'z',Vis_lay.vis_lay));
	}
for (auto i:alpha  )
{
	M(i.state_vector,0)+=i.co_efficient;
}
M(beta.state_vector,0)+=beta.co_efficient;
return M;
	
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







neural_net::some_visible_layer neural_net::some_visible_layer::operator+(some_visible_layer  alpha)
{
	if (alpha.state_vector==state_vector)
	{
		some_visible_layer m(alpha.state_vector,alpha.co_efficient+co_efficient);
		return m;
	}
	else
	{
		std::cerr<<"error the states are diff and hence do not go together well \n";
	}
}
neural_net::some_visible_layer neural_net::some_visible_layer::operator*(some_visible_layer  alpha)
{
	if (alpha.state_vector==state_vector)
	{
		some_visible_layer m(alpha.state_vector,alpha.co_efficient*co_efficient);
		return m;
	}
	else
	{
		std::cerr<<"error the states are diff and hence do not go together well \n";
	}
}
#endif