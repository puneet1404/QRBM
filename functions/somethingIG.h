
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

	class Neural_net
	{
	private:
		vector<matrix> nodes;

		double B_value[alpha];
		matrix Hidden_layer_init()
		{
			matrix Hidden_layer = arma::randu(alpha,columns);
			return Hidden_layer;
		}

	public:
	};
}
#endif