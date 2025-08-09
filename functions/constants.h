#ifndef _constants_H_
#define _constants_H_

#include "somethingig2.h"

// constantants for the vmc;
namespace pj
{
    // spin lattice properties
    int col = 1;
    int row = 10;
    // interaction variables
    long double H = 0.5;
    long double J = 1;
    // neural network parameters
    int alpha = 4;
    int hid_node_num = alpha * row;

    // training parameters
    double gama_init_value = .1;
    double gama_decrement_exponent = 0.8; // 0.3 is one of the optimal values converges in like 200o turns
    double itt_value = 1000;

    // quality of life
    bool picture_rest = false;
    bool display_togle = true;

    // running average window
    int run_avg_win = 10;

    //compute exact results or not
    bool exact_cal_bool = 1;


} // namespace pj

#endif
