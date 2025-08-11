#ifndef _constants_H_
#define _constants_H_

#include <cmath>
#include "RBM.h"
#include "ActivationFunction.h"
// constantants for the vmc;
namespace pj
{
    // spin lattice properties
    const int col = 1;
    const int row = 2;
    // interaction variables
    const long double H = 1;
    const long double J = 1;
    // neural network parameters
    const int alpha = 4;
    const int hid_node_num = alpha * row;

    // training parameters
    const double gama_init_value = .1;
    const double gama_decrement_exponent = 1;
    const double itt_value = 1000;

    // activation functions
    double (*activation_function)(double) = lin_max;
    double (*activation_function_derivative)(double) = d_lin_max;

    // quality of life
    const bool picture_rest = false;
    const bool display_togle = true;
    const int graph_cuttoff = 0;
    const bool graph_clear_after_interval(true);
    const int graph_clear_interval(1000);


    // running average window
    const int run_avg_win = 20;
    const int plot_interval = 1000;

    // compute exact results or not

    const bool exact_cal_bool = (row > 10) ? (false) : (true);


} // namespace pj

#endif
