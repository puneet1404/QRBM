#ifndef _somethingig_H_
#define _somethingig_H_

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

const int rows = 1;
const int columns = 3;
const int alpha = 4;

typedef arma::mat matrix;
namespace neural_net
{
	// just a thing to call visible layer
	struct visible_layer
	{
		matrix vis_lay = arma::randu(columns, 1);
		matrix state_vector = arma::zeros(pow(2, columns), 1);
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
		int n = 1;

	public:
		double E_loc();
		matrix &visible_layer();
		matrix b = arma::randu(alpha, 1);
		matrix a = arma::randu(columns, 1);
		matrix hidden_layer;
		double psi_s();
		double psi_s(matrix &S);
		Neural_net();
	};
}

// implimentation
matrix &neural_net::Neural_net::visible_layer()
{
	return Vis_lay.vis_lay;
}

neural_net::Neural_net::Neural_net()
{
	Hidden_layer_init();
}
void neural_net::Neural_net::Hidden_layer_init()
{
	hidden_layer = arma::randu(alpha, columns);
	b = arma::randu(b.n_rows, b.n_cols);
	Vis_lay.state_vector_init();
}

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







//implimnetation for visible_layer struct 
void neural_net::visible_layer::state_vector_init()
{
	int n=0;
	vis_lay.for_each([](matrix::elem_type &m)
							 { (m > 0.5) ? (m = -1) : (m = 1); });
	for(auto i:vis_lay){
		(i==1)?((n=n<<1)):((n=(n<<1)+1));
	}
	state_vector(n,0)=1;
	cout<<state_vector<<"\n";

}
#endif