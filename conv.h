#include <hls_half.h>
#include <iostream>
#include "ap_axi_sdata.h"
#include <cmath>

#ifndef _CONV_H_
#define _CONV_H_

#define R_MAX 32
#define C_MAX 32
#define CHin_MAX 128
#define CHout_MAX 32
#define K_MAX 3
#define CHin_tiny 128
#define CHout_tiny 32

typedef half feature_map;
typedef half weight_type;

struct axis_t {
    half data;
    ap_int<1> last;
};

void top(axis_t* In_ddr, axis_t* W_ddr, axis_t* B_ddr, axis_t* Out_m_ddr,axis_t* In_ddr_2, axis_t* W_ddr_2, axis_t* B_ddr_2, axis_t* Out_m_ddr_2, int ctr);

#endif
