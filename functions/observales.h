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
            // static weights check_w;
            // cout<<!check_w.are_equal(*w)<<endl; 
            // if (!check_w.are_equal(*w))
            {
                Z = 0;
                set<unsigned long int> hash;
                visible_layer VL = *vl;
                for (size_t i = 0; i < itt_value; i++)
                {
                    hash.insert(VL.to_int());
                    VL = sampler(VL, *w, pj::rd);
                }
                for (auto i = hash.begin(); i != hash.end(); i++)
                {
                    VL.to_S(*i);
                    Z += psi(VL, *w);
                }
                // check_w=*w;
            }

            return Z;
        }
        double mag_x(visible_layer vis_lay,function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler, double z = 0)
        {
            // if (z = 0)
            // z = calc_z(sampler);
            double m_x = 0;
            visible_layer vl_m = vis_lay;
            for (size_t i = 0; i < row; i++)
            {
                vl_m.flip(i);
                // m_x += psi(*vl, *w) * psi(vl_m, *w) / (z * z);
                m_x+=p_ratio_fast(i,*vl,*w);
                vl_m.flip(i);
            }
            return m_x;
        }
        double mag_x_avg(function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler)
        {
            calc_z(sampler);
            double mag_x_av = 0;
            visible_layer vis_lay=*vl;
            for (size_t i = 0; i < itt_value; i++)
            {
                mag_x_av += mag_x(vis_lay,sampler, Z);
                vis_lay=sampler(vis_lay,*w,rd);
            }
            mag_x_av /= itt_value;
            return mag_x_av;
        }
    };

} // namespace pj

#endif