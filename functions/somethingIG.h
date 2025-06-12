
#ifndef _somethinglG_H_
#define _somethinglG_H_
#include<eigen3/Eigen/Dense>
#include<armadillo>
#include<vector>
#include<iostream>
using namespace std;

const int rows=1;
const int columns =5;
const int alpha= 4;

class Hidden_Layer
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
	Hidden_Layer()
	{
		for (int i = 0; i < alpha; i++)
		{
			nodes.push_back(Hidden_layer_init());
			B_value[i] = rand() / RAND_MAX;
		}
	}
	Eigen::Matrix<double, columns, rows> &operator()(int i)
	{
		return nodes[i];
	}
};
#endif