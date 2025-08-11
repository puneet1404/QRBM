#ifndef ActivationFunction_H
#define ActivationFunction_H

#include <cmath>
#include "constants.h"
namespace pj
{
    double Relu_factor = 1;
    double lin_max_limit = 1;

    double sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }
    double d_sigmoid(double x)
    {
        return (1 - sigmoid(x)) * sigmoid(x);
    }

    double d_tanh(double x)
    {
        return 1 - pow(tanh(x), 2);
    }

    double d_atan(double x)
    {
        return (1 / (pow(x, 2) + 1));
    }

    double Relu(double x)
    {
        return (x < 0) ? (Relu_factor * x) : (x);
    }
    double d_Relu(double x)
    {
        return (x < 0) ? (Relu_factor) : (1);
    }

    double linear(double x)
    {
        return x;
    }
    double d_linear(double x)
    {
        return 1;
    }

    double log_cosh(double x)
    {
        return log(cosh(x));
    }
    double d_log_cosh(double x)
    {
        return (tanh(x));
    }

    double lin_max(double x)
    {
        return (abs(x) < lin_max_limit) ? (x) : (lin_max_limit);
    }
    double d_lin_max(double x)
    {
        return (abs(x) < lin_max_limit) ? (1) : (0);
    }
}
#endif