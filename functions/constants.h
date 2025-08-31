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
    const int row = 20;
    // interaction variables
    const long double H = 1;
    const long double J = 1;
    // neural network parameters
    const int alpha = 2;
    const int hid_node_num = alpha * row;

    // training parameters
    const double gama_init_value = .1;
    const double gama_decrement_exponent = 1;
    const double mean = 0; // mean for initial random variable
    const double sd = 0.001; // sd fro normal distribution of initial variable
    const double itt_value =6000;
    const bool multi_check = false;
    const bool check_mulitple_vales_of_update = false;
    const int check_mulitple_vales_of_update_after = 5000;
    const int no_of_mulitple_vales_of_update = 10;
    double beta =1;

    // ! activation funtion for now are not being used
    // activation functions
    double (*activation_function)(double) = sigmoid;
    double (*activation_function_derivative)(double) = d_sigmoid;

    //magnetization partition function sampler 
    // visible_layer (*magnetization_sampler)(visible_layer , const weights &, std::random_device &rd) =sampler_md ;

    // quality of life
    const bool picture_rest = false;
    const bool display_togle = true;
    const int graph_cuttoff = 0;
    const bool graph_clear_after_interval(false);
    const int graph_clear_interval(0);

    // running average window
    const int run_avg_win = 50;
    const int plot_interval = 50;

    // compute exact results or not

    const bool exact_cal_bool = (row > 10) ? (false) : (true);

} // namespace pj

#endif
