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
    const int row = 10;
    // interaction variables
    const long double H = 1;
    const long double J = 1;
    // neural network parameters
    const int alpha = 1;
    const int hid_node_num = alpha * row;

    // training parameters
    const double gama_init_value = .1;
    const double gama_decrement_exponent = 1;
    const double itt_value = 1000;
    const bool check_mulitple_vales_of_update = false;
    const int no_of_mulitple_vales_of_update = 10;

    // activation functions
    double (*activation_function)(double) = linear;
    double (*activation_function_derivative)(double) = d_linear;

    // quality of life
    const bool picture_rest = true;
    const bool display_togle = true;
    const int graph_cuttoff = 250;
    const bool graph_clear_after_interval(false);
    const int graph_clear_interval(0);


    // running average window
    const int run_avg_win = 50;
    const int plot_interval = 10;

    // compute exact results or not

    const bool exact_cal_bool = (row > 10) ? (false) : (true);


} // namespace pj

#endif
