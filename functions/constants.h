#ifndef _constants_H_
#define _constants_H_

#include "somethingig2.h"

// constantants for the vmc;
namespace pj
{
    int col = 1;
    int row = 7;
    int alpha = 1;
    int hid_node_num = alpha * row;
    long double H = 1.1;
    long double J = 1;

    //
    double gama_init_value = .01;
    double gama_decrement_exponent = 1;
    double itt_value = 1000;

    //
    bool picture_rest = false;

} // namespace pj

#endif
