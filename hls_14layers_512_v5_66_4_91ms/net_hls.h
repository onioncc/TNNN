#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>



//#define CSIM_DEBUG
#define CSIM_DEBUG_FIX


typedef ap_uint<8> uint8;
//typedef ap_uint<16> uint16;
typedef ap_uint<4> uint4;




#ifdef CSIM_DEBUG
	typedef float FIX_32_4;	//fix point
	typedef float FIX_32_25;	//fix point
	typedef float FIX_FM;	//fix point for feature map
	typedef float FIX_FM_last;
	typedef float FIX_WT;	//fix point for weights
	typedef float FIX_32_16;
	typedef float FIX_32_10;
	typedef float FIX_32_12;
	typedef float FIX_16_6;
	typedef float FIX_16_5;
	typedef float FIX_16_4;
	typedef float FIX_16_10;
	typedef float uint128;
#else
	//typedef ap_fixed<16, 6, AP_RND_INF, AP_SAT> FIX_FM;	//fix point for feature map
	typedef ap_fixed<8, 3, AP_RND, AP_SAT> FIX_FM;	//fix point for feature map
	typedef ap_fixed<10,4, AP_RND, AP_SAT> FIX_WT;	//fix point for weights
	typedef ap_fixed<16, 6, AP_RND, AP_SAT> FIX_16_6;
	typedef ap_fixed<16, 5, AP_RND, AP_SAT> FIX_16_5;
	typedef ap_fixed<16, 4, AP_RND, AP_SAT> FIX_16_4;
	typedef ap_fixed<16, 3, AP_RND, AP_SAT> FIX_16_3;
	typedef ap_fixed<16, 10, AP_RND, AP_SAT> FIX_16_10;
	typedef ap_fixed<32,16, AP_RND, AP_SAT> FIX_32_16;
	typedef ap_fixed<32,12, AP_RND, AP_SAT> FIX_32_12;
	typedef ap_fixed<32,10, AP_RND, AP_SAT> FIX_32_10;
	typedef ap_fixed<32, 4, AP_RND, AP_SAT> FIX_32_4;
	typedef ap_fixed<32, 7, AP_RND, AP_SAT> FIX_32_7;
	typedef ap_fixed<32,25, AP_RND, AP_SAT> FIX_32_25;
	typedef ap_uint<128> uint128;
	//typedef ap_uint<8> uint128;
#endif






// 48, 96, 192, 384, 512
// if cannot be divided by BUF_DPTH, need to plus one

#define BUF_DPTH_32

#ifdef BUF_DPTH_32


#define BUF_DPTH		32
#define PAR_FACT		16


#define l1_cd	1
#define l1_rd	1
#define l1_bd	1

#define	l2_cd	(48 / BUF_DPTH + 1)
#define	l2_rd	(48 / BUF_DPTH + 1) * (BUF_DPTH / BUF_DPTH)
#define	l2_bd	(48 / BUF_DPTH + 1)

#define	l3_cd	(48 / BUF_DPTH + 1)
#define	l3_rd	(48 / BUF_DPTH + 1)
#define	l3_bd	(48 / BUF_DPTH + 1)

#define	l4_cd	(48 / BUF_DPTH + 1)
#define	l4_rd	(48 / BUF_DPTH + 1)
#define	l4_bd	(48 / BUF_DPTH + 1)

#define	l5_cd	(96 / BUF_DPTH)
#define	l5_rd	(96 / BUF_DPTH) * (48 / BUF_DPTH + 1)
#define	l5_bd	(96 / BUF_DPTH)

#define	l6_cd	(96 / BUF_DPTH)
#define	l6_rd	(96 / BUF_DPTH)
#define	l6_bd	(96 / BUF_DPTH)

#define	l7_cd	(96 / BUF_DPTH)
#define	l7_rd	(96 / BUF_DPTH)
#define	l7_bd	(96 / BUF_DPTH)

#define l8_cd	((192 / BUF_DPTH))
#define l8_rd	((192 / BUF_DPTH) * (96 / BUF_DPTH))
#define l8_bd	(192 / BUF_DPTH)

#define	l9_cd	(192 / BUF_DPTH)
#define	l9_rd	(192 / BUF_DPTH)
#define	l9_bd	(192 / BUF_DPTH)

#define	l10_cd	(192 / BUF_DPTH)
#define	l10_rd	(192 / BUF_DPTH)
#define	l10_bd	(192 / BUF_DPTH)

#define l11_cd	((384 / BUF_DPTH))
#define l11_rd	((384 / BUF_DPTH) * (192 / BUF_DPTH))
#define l11_bd	(384 / BUF_DPTH)

#define	l12_cd	(384 / BUF_DPTH)
#define	l12_rd	(384 / BUF_DPTH)
#define	l12_bd	(384 / BUF_DPTH)

#define l13_cd	((512 / BUF_DPTH))
#define l13_rd	((512 / BUF_DPTH) * (384 / BUF_DPTH))
#define l13_bd	(512 / BUF_DPTH)

#define l14_cd	(10 / BUF_DPTH + 1)
#define l14_rd	(512 / BUF_DPTH)

#endif




#ifdef BUF_DPTH_24


#define BUF_DPTH		24
#define PAR_FACT		12


#define l1_cd	1
#define l1_rd	1
#define l1_bd	1

#define	l2_cd	(48 / BUF_DPTH)
#define	l2_rd	(48 / BUF_DPTH) * (BUF_DPTH / BUF_DPTH)
#define	l2_bd	(48 / BUF_DPTH)

#define	l3_cd	(48 / BUF_DPTH)
#define	l3_rd	(48 / BUF_DPTH)
#define	l3_bd	(48 / BUF_DPTH)

#define	l4_cd	(48 / BUF_DPTH)
#define	l4_rd	(48 / BUF_DPTH)
#define	l4_bd	(48 / BUF_DPTH)

#define	l5_cd	(96 / BUF_DPTH)
#define	l5_rd	(96 / BUF_DPTH) * (48 / BUF_DPTH)
#define	l5_bd	(96 / BUF_DPTH)

#define	l6_cd	(96 / BUF_DPTH)
#define	l6_rd	(96 / BUF_DPTH)
#define	l6_bd	(96 / BUF_DPTH)

#define	l7_cd	(96 / BUF_DPTH)
#define	l7_rd	(96 / BUF_DPTH)
#define	l7_bd	(96 / BUF_DPTH)

#define l8_cd	((192 / BUF_DPTH))
#define l8_rd	((192 / BUF_DPTH) * (96 / BUF_DPTH))
#define l8_bd	(192 / BUF_DPTH)

#define	l9_cd	(192 / BUF_DPTH)
#define	l9_rd	(192 / BUF_DPTH)
#define	l9_bd	(192 / BUF_DPTH)

#define	l10_cd	(192 / BUF_DPTH)
#define	l10_rd	(192 / BUF_DPTH)
#define	l10_bd	(192 / BUF_DPTH)

#define l11_cd	((384 / BUF_DPTH))
#define l11_rd	((384 / BUF_DPTH) * (192 / BUF_DPTH))
#define l11_bd	(384 / BUF_DPTH)

#define	l12_cd	(384 / BUF_DPTH)
#define	l12_rd	(384 / BUF_DPTH)
#define	l12_bd	(384 / BUF_DPTH)

#define l13_cd	((512 / BUF_DPTH) + 1)
#define l13_rd	((512 / BUF_DPTH + 1) * (384 / BUF_DPTH))
#define l13_bd	(512 / BUF_DPTH + 1)

#define l14_cd	(10 / BUF_DPTH + 1)
#define l14_rd	(512 / BUF_DPTH + 1)

#endif



#ifdef BUF_DPTH_16


#define BUF_DPTH		16
#define PAR_FACT		16


#define l1_cd	1
#define l1_rd	1
#define l1_bd	1

#define	l2_cd	(48 / BUF_DPTH)
#define	l2_rd	(48 / BUF_DPTH) * (BUF_DPTH / BUF_DPTH)
#define	l2_bd	(48 / BUF_DPTH)

#define	l3_cd	(48 / BUF_DPTH)
#define	l3_rd	(48 / BUF_DPTH)
#define	l3_bd	(48 / BUF_DPTH)

#define	l4_cd	(48 / BUF_DPTH)
#define	l4_rd	(48 / BUF_DPTH)
#define	l4_bd	(48 / BUF_DPTH)

#define	l5_cd	(96 / BUF_DPTH)
#define	l5_rd	(96 / BUF_DPTH) * (48 / BUF_DPTH)
#define	l5_bd	(96 / BUF_DPTH)

#define	l6_cd	(96 / BUF_DPTH)
#define	l6_rd	(96 / BUF_DPTH)
#define	l6_bd	(96 / BUF_DPTH)

#define	l7_cd	(96 / BUF_DPTH)
#define	l7_rd	(96 / BUF_DPTH)
#define	l7_bd	(96 / BUF_DPTH)

#define l8_cd	((192 / BUF_DPTH))
#define l8_rd	((192 / BUF_DPTH) * (96 / BUF_DPTH))
#define l8_bd	(192 / BUF_DPTH)


#define	l9_cd	(192 / BUF_DPTH)
#define	l9_rd	(192 / BUF_DPTH)
#define	l9_bd	(192 / BUF_DPTH)


#define	l10_cd	(192 / BUF_DPTH)
#define	l10_rd	(192 / BUF_DPTH)
#define	l10_bd	(192 / BUF_DPTH)

#define l11_cd	((384 / BUF_DPTH))
#define l11_rd	((384 / BUF_DPTH) * (192 / BUF_DPTH))
#define l11_bd	(384 / BUF_DPTH)

#define	l12_cd	(384 / BUF_DPTH)
#define	l12_rd	(384 / BUF_DPTH)
#define	l12_bd	(384 / BUF_DPTH)

#define l13_cd	((512 / BUF_DPTH))
#define l13_rd	((512 / BUF_DPTH) * (384 / BUF_DPTH))
#define l13_bd	(512 / BUF_DPTH)

#define l14_cd	(10 / BUF_DPTH + 1)
#define l14_rd	(512 / BUF_DPTH)

#endif

//3+18+72+288+768+32
#define	all_1x1		l2_rd + l5_rd + l8_rd + l11_rd + l13_rd + l14_rd

//1+3+6+12+24
#define all_3x3		l1_rd + l4_rd + l7_rd + l10_rd + l12_rd

//1+3+3+6+6+12+12+24+24+32
#define all_bias	l1_bd + l2_bd + l4_bd + l5_bd + l7_bd + l8_bd + l10_bd + l11_bd + l12_bd + l13_bd





void mobilenet(uint8 image_in_raw_pad[3][162][322],

				uint128 fix_conv_weight_1x1_all_128bit[all_1x1][BUF_DPTH/8][BUF_DPTH],
				uint128 fix_conv_weight_3x3_all_128bit[all_3x3][BUF_DPTH/8][3][3],
				uint128 fix_bias_all[all_bias][BUF_DPTH/8],

				uint128 DDR_pool_3_out_PL[l3_cd * BUF_DPTH][82][162],
				uint128 DDR_pool_6_out_PL[l6_cd * BUF_DPTH][42][82],

				uint128 DDR_buf[36][BUF_DPTH][22][42],

				float predict_box[5]

);

void CONV_3x3_group(FIX_FM bottom[BUF_DPTH][22][42],
					FIX_FM top[BUF_DPTH][22][42],
					FIX_WT weight[BUF_DPTH][3][3],
					FIX_WT bias[BUF_DPTH][1],
					int skip);

void CONV_1x1(FIX_FM bottom[BUF_DPTH][22][42],
			  FIX_FM top[BUF_DPTH][22][42],
			  FIX_WT weights[BUF_DPTH][BUF_DPTH],
			  uint4 mode);



