#ifndef _observales_H_
#define _observales_H_

#include"RBM.h"

namespace pj
{
    struct magnetization
    {
        visible_layer* vl = nullptr; 
        weights* w = nullptr;
        double Z=0;
        magnetization(visible_layer* VL, weights* W)
        {
            vl=VL;
            w=W;
        }
        void clac_z(function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler )
        {
            visible_layer VL =*vl;
            for (size_t i = 0; i < itt_value; i++)
            {
                Z+= psi(*vl,*w);
            }
            Z/=itt_value;
            Z*=pow(2,row);
        }

    
    };
    
} // namespace pj


#endif