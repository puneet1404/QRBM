#ifndef _observales_H_
#define _observales_H_

#include "RBM.h"

namespace pj
{
    struct magnetization
    {
        visible_layer *vl = nullptr;
        weights *w = nullptr;
        double Z = 0;
        magnetization(visible_layer *VL, weights *W)
        {
            vl = VL;
            w = W;
        }
        double calc_z(function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler)
        {
            Z = 0;
            set<visible_layer> S;
            visible_layer VL = *vl;
            for (size_t i = 0; i < itt_value; i++)
            {
                // S.insert(VL);
                VL = sampler(VL, *w, pj::rd);
            }
            // for (auto i = S.begin(); i==S.end(); i++)
            // {
            //     Z+=psi(*i,*w);
            // }
            cout<<"z="<<Z<<"\n";
            return Z;
        }
        double mag_x(function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler, double z = 0)
        {
            // if (z = 0)
                // z = calc_z(sampler);
            double m_x = 0;
            visible_layer vl_m = *vl;
            for (size_t i = 0; i < row; i++)
            {
                vl_m.flip(i);
                m_x += psi(*vl, *w) * psi(vl_m, *w) / (z * z);
                vl_m.flip(i);
            }
            return m_x;
        }
        double mag_x_avg(function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler)
        {
            calc_z(sampler);
            double mag_x_av = 0;
            for (size_t i = 0; i < itt_value; i++)
            {
                mag_x_av += mag_x(sampler, Z);
            }
            mag_x_av /= itt_value;
            return mag_x_av;
        }
    };

} // namespace pj

#endif