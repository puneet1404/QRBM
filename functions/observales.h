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
        function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler;
        magnetization(visible_layer *VL, weights *W, function<visible_layer(const visible_layer, const weights &, std::random_device &)> sampler_in)
        {
            sampler = sampler_in;
            vl = VL;
            w = W;
        }
        double mag_x(visible_layer vis_lay, double z = 0)
        {
            // if (z = 0)
            // z = calc_z(sampler);
            double m_x = 0;
            visible_layer vl_m = vis_lay;
            for (size_t i = 0; i < row; i++)
            {
                int n = vl_m.to_int();
                vl_m.flip(i);
                if (vl_m.to_int() == n)
                    throw std::runtime_error("flipping not happening ");
                m_x += norm(p_ratio(vl_m, *vl, *w));
                vl_m.flip(i);
            }
            return m_x;
        }
        double mag_part_x(int part_no = 5)
        {
            double mag_cal = 0;
            visible_layer vl_f = *vl, vl_i=*vl;

            for (int n = row * 2 / part_no; n < row * 3 / part_no; n++)
            {
                vl_f.flip(n);

                mag_cal += norm(p_ratio(vl_f, vl_i, *w));

                vl_f.flip(n);
            }
            return mag_cal;
        }
        double mag_x_avg()
        {
            // calc_z();
            double mag_x_av = 0;
            // visible_layer vis_lay = *vl;
            for (size_t i = 0; i < itt_value; i++)
            {
                mag_x_av += mag_x(*vl, Z);
                *vl = sampler(*vl, *w, rd);
            }
            mag_x_av = mag_x_av / itt_value;
            return mag_x_av;
        }
        double mag_part_x_avg()
        {
            double mag_x_av = 0;
            // visible_layer vis_lay = *vl;
            for (size_t i = 0; i < itt_value; i++)
            {
                mag_x_av += mag_part_x();
                *vl = sampler(*vl, *w, rd);
            }
            mag_x_av = mag_x_av / itt_value;
            return mag_x_av;
        }
    };

} // namespace pj

#endif