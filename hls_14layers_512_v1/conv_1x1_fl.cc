

// Conv 1x1 PE

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "net_hls.h"


FIX_FM compute_engine_16(FIX_WT w0,  FIX_FM b0,
					  FIX_WT w1,  FIX_FM b1,
					  FIX_WT w2,  FIX_FM b2,
					  FIX_WT w3,  FIX_FM b3,
					  FIX_WT w4,  FIX_FM b4,
					  FIX_WT w5,  FIX_FM b5,
					  FIX_WT w6,  FIX_FM b6,
					  FIX_WT w7,  FIX_FM b7,
					  FIX_WT w8,  FIX_FM b8,
					  FIX_WT w9,  FIX_FM b9,
					  FIX_WT w10, FIX_FM b10,
					  FIX_WT w11, FIX_FM b11,
					  FIX_WT w12, FIX_FM b12,
					  FIX_WT w13, FIX_FM b13,
					  FIX_WT w14, FIX_FM b14,
					  FIX_WT w15, FIX_FM b15)
{
	//#pragma HLS ALLOCATION instances=mul limit=8 operation

	FIX_32_10 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	FIX_32_10 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	FIX_32_10 add0, add1, add2, add3,  add4,  add5,  add6;
	FIX_32_10 add7, add8, add9, add10, add11, add12, add13, add14;

	mul0  = (FIX_16_5)w0  * (FIX_16_5)b0;
	mul1  = (FIX_16_5)w1  * (FIX_16_5)b1;
	mul2  = (FIX_16_5)w2  * (FIX_16_5)b2;
	mul3  = (FIX_16_5)w3  * (FIX_16_5)b3;
	mul4  = (FIX_16_5)w4  * (FIX_16_5)b4;
	mul5  = (FIX_16_5)w5  * (FIX_16_5)b5;
	mul6  = (FIX_16_5)w6  * (FIX_16_5)b6;
	mul7  = (FIX_16_5)w7  * (FIX_16_5)b7;
	mul8  = (FIX_16_5)w8  * (FIX_16_5)b8;
	mul9  = (FIX_16_5)w9  * (FIX_16_5)b9;
	mul10 = (FIX_16_5)w10 * (FIX_16_5)b10;
	mul11 = (FIX_16_5)w11 * (FIX_16_5)b11;
	mul12 = (FIX_16_5)w12 * (FIX_16_5)b12;
	mul13 = (FIX_16_5)w13 * (FIX_16_5)b13;
	mul14 = (FIX_16_5)w14 * (FIX_16_5)b14;
	mul15 = (FIX_16_5)w15 * (FIX_16_5)b15;


	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;
	add4 = mul8  + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8  = add0 + add1;
	add9  = add2 + add3;
	add10 = add4 + add5;
	add11 = add6 + add7;

	add12 = add8  + add9;
	add13 = add10 + add11;

	add14 = add12 + add13;

	return add14;

}



FIX_FM compute_engine_8(FIX_WT w0,  FIX_FM b0,
					  FIX_WT w1,  FIX_FM b1,
					  FIX_WT w2,  FIX_FM b2,
					  FIX_WT w3,  FIX_FM b3,
					  FIX_WT w4,  FIX_FM b4,
					  FIX_WT w5,  FIX_FM b5,
					  FIX_WT w6,  FIX_FM b6,
					  FIX_WT w7,  FIX_FM b7)
{
	FIX_32_16 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	FIX_32_16 add0, add1, add2, add3,  add4,  add5,  add6;

	mul0  = w0  * b0;
	mul1  = w1  * b1;
	mul2  = w2  * b2;
	mul3  = w3  * b3;
	mul4  = w4  * b4;
	mul5  = w5  * b5;
	mul6  = w6  * b6;
	mul7  = w7  * b7;

	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;

	add4 = add0  + add1;
	add5 = add2 + add3;

	add6 = add4 + add5;

	return (FIX_FM)add6;

}


/*

void CONV_1x1(FIX_FM bottom[BUF_DPTH][22][42],
			  FIX_FM top[BUF_DPTH][22][42],
			  FIX_WT weights[BUF_DPTH][BUF_DPTH])
{
FIX_WT weight_buf[BUF_DPTH][BUF_DPTH];

//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete

#pragma HLS array_partition variable=top dim=1 cyclic factor=16
#pragma HLS array_partition variable=bottom dim=1 cyclic factor=16

#pragma HLS array_partition variable=weight_buf dim=1 complete
#pragma HLS array_partition variable=weight_buf dim=2 complete

//#pragma HLS ALLOCATION instances=compute_engine_16 limit=8 function

	for(int i = 0; i < BUF_DPTH; i++)
		for(int j = 0; j < BUF_DPTH; j++)
			weight_buf[i][j] = weights[i][j];


	for(int h = 1; h <= 20; h++){
		for(int w = 1; w <= 40; w++) {

			for(int cin = 0; cin < BUF_DPTH; cin+=16) {
#pragma HLS pipeline
				for(int co = 0; co < BUF_DPTH; co+=8) {
					for(int coo = 0; coo < 8; coo++) {
#pragma HLS unroll
						top[co+coo][h][w] += compute_engine_16(
												 weight_buf[co+coo][0 + cin],   bottom[0 + cin][h][w],
												 weight_buf[co+coo][1 + cin],   bottom[1 + cin][h][w],
												 weight_buf[co+coo][2 + cin],   bottom[2 + cin][h][w],
												 weight_buf[co+coo][3 + cin],   bottom[3 + cin][h][w],
												 weight_buf[co+coo][4 + cin],   bottom[4 + cin][h][w],
												 weight_buf[co+coo][5 + cin],   bottom[5 + cin][h][w],
												 weight_buf[co+coo][6 + cin],   bottom[6 + cin][h][w],
												 weight_buf[co+coo][7 + cin],   bottom[7 + cin][h][w],
												 weight_buf[co+coo][8 + cin],   bottom[8 + cin][h][w],
												 weight_buf[co+coo][9 + cin],   bottom[9 + cin][h][w],
												 weight_buf[co+coo][10 + cin],  bottom[10 + cin][h][w],
												 weight_buf[co+coo][11 + cin],  bottom[11 + cin][h][w],
												 weight_buf[co+coo][12 + cin],  bottom[12 + cin][h][w],
												 weight_buf[co+coo][13 + cin],  bottom[13 + cin][h][w],
												 weight_buf[co+coo][14 + cin],  bottom[14 + cin][h][w],
												 weight_buf[co+coo][15 + cin],  bottom[15 + cin][h][w]);
					}
				}
			}
		}
	}
}

*/




void CONV_1x1(FIX_FM bottom[BUF_DPTH][22][42],
			  FIX_FM top[BUF_DPTH][22][42],
			  FIX_WT weights[BUF_DPTH][BUF_DPTH])
{
FIX_WT weight_buf[BUF_DPTH][BUF_DPTH];

//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete

#pragma HLS array_partition variable=top dim=1 cyclic factor=16
#pragma HLS array_partition variable=bottom dim=1 cyclic factor=16

#pragma HLS array_partition variable=weight_buf dim=1 complete
#pragma HLS array_partition variable=weight_buf dim=2 complete

//#pragma HLS ALLOCATION instances=compute_engine_16 limit=8 function

FIX_FM tmp[32];
#pragma HLS array_partition variable=tmp dim=1 complete


	for(int i = 0; i < BUF_DPTH; i++)
		for(int j = 0; j < BUF_DPTH; j++)
			weight_buf[i][j] = weights[i][j];


	for(int h = 1; h <= 20; h++){
		for(int w = 1; w <= 40; w++) {

			for(int cin = 0; cin < BUF_DPTH; cin+=16) {
#pragma HLS pipeline II=2
				for(int co = 0; co < 16; co++) {
#pragma HLS unroll
					tmp[co] = compute_engine_16(
											 weight_buf[co][0 + cin],   bottom[0 + cin][h][w],
											 weight_buf[co][1 + cin],   bottom[1 + cin][h][w],
											 weight_buf[co][2 + cin],   bottom[2 + cin][h][w],
											 weight_buf[co][3 + cin],   bottom[3 + cin][h][w],
											 weight_buf[co][4 + cin],   bottom[4 + cin][h][w],
											 weight_buf[co][5 + cin],   bottom[5 + cin][h][w],
											 weight_buf[co][6 + cin],   bottom[6 + cin][h][w],
											 weight_buf[co][7 + cin],   bottom[7 + cin][h][w],
											 weight_buf[co][8 + cin],   bottom[8 + cin][h][w],
											 weight_buf[co][9 + cin],   bottom[9 + cin][h][w],
											 weight_buf[co][10 + cin],  bottom[10 + cin][h][w],
											 weight_buf[co][11 + cin],  bottom[11 + cin][h][w],
											 weight_buf[co][12 + cin],  bottom[12 + cin][h][w],
											 weight_buf[co][13 + cin],  bottom[13 + cin][h][w],
											 weight_buf[co][14 + cin],  bottom[14 + cin][h][w],
											 weight_buf[co][15 + cin],  bottom[15 + cin][h][w]);
				}

				for(int co = 0; co < 16; co++) {
#pragma HLS unroll
					top[co][h][w] += tmp[co];
				}
			}
		}
	}




	for(int h = 1; h <= 20; h++){
		for(int w = 1; w <= 40; w++) {

			for(int cin = 0; cin < BUF_DPTH; cin+=16) {
#pragma HLS pipeline II=2
				for(int co = 16; co < 32; co++) {
#pragma HLS unroll
					tmp[co] = compute_engine_16(
											 weight_buf[co][0 + cin],   bottom[0 + cin][h][w],
											 weight_buf[co][1 + cin],   bottom[1 + cin][h][w],
											 weight_buf[co][2 + cin],   bottom[2 + cin][h][w],
											 weight_buf[co][3 + cin],   bottom[3 + cin][h][w],
											 weight_buf[co][4 + cin],   bottom[4 + cin][h][w],
											 weight_buf[co][5 + cin],   bottom[5 + cin][h][w],
											 weight_buf[co][6 + cin],   bottom[6 + cin][h][w],
											 weight_buf[co][7 + cin],   bottom[7 + cin][h][w],
											 weight_buf[co][8 + cin],   bottom[8 + cin][h][w],
											 weight_buf[co][9 + cin],   bottom[9 + cin][h][w],
											 weight_buf[co][10 + cin],  bottom[10 + cin][h][w],
											 weight_buf[co][11 + cin],  bottom[11 + cin][h][w],
											 weight_buf[co][12 + cin],  bottom[12 + cin][h][w],
											 weight_buf[co][13 + cin],  bottom[13 + cin][h][w],
											 weight_buf[co][14 + cin],  bottom[14 + cin][h][w],
											 weight_buf[co][15 + cin],  bottom[15 + cin][h][w]);
				}

				for(int co = 16; co < 32; co++) {
#pragma HLS unroll
					top[co][h][w] += tmp[co];
				}
			}
		}
	}
}

