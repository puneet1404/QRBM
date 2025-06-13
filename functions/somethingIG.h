
#ifndef _somethinglG_H_
#define _somethinglG_H_
#include <armadillo>
#include <vector>
#include <iostream>
using namespace std;

const int rows = 1;
const int columns = 5;
const int alpha = 4;

typedef arma::mat matrix;
namespace neural_net
{

	// a class to simmulate the nodes(number is alpha) and the input 1d chain is column
	class Neural_net
	{
	private:
	//this is the matrix which is multiplied to the column vector for the node values to be obtained 
		vector<matrix> nodes;

		double B_value[alpha];
		matrix Hidden_layer_init();
	public:
	Neural_net();
	};
}




















//implimentation
	matrix neural_net::Neural_net::Hidden_layer_init()
		{
			matrix Hidden_layer = arma::randu(alpha,columns);
			return Hidden_layer;
		}

#endif