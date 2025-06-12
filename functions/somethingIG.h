
#ifndef _somethinglG_H_
#define _somethinglG_H_
#include <eigen3/Eigen/Dense>
#include <armadillo>
#include <vector>
#include <iostream>
using namespace std;

const int rows = 1;
const int columns = 5;
const int alpha = 4;

namespace neural_net
{

	class Neural_net
	{
	private:
		vector<Eigen::Matrix<double, columns, rows>> nodes;

		double B_value[alpha];
		Eigen::MatrixXd Hidden_layer_init()
		{
			Eigen::MatrixXd Hidden_layer = Eigen::MatrixXd::Random(columns, rows);
			return Hidden_layer;
		}

	public:
	};
}
#endif