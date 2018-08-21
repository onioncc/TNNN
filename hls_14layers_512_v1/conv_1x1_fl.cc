

// Conv 1x1 PE

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "net_hls.h"


FIX_32_12 compute_engine_16(FIX_WT w0,  FIX_FM b0,
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
//	#pragma HLS ALLOCATION instances=mul limit=8 operation

	FIX_32_12 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	FIX_32_12 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	FIX_32_12 add0, add1, add2, add3,  add4,  add5,  add6;
	FIX_32_12 add7, add8, add9, add10, add11, add12, add13, add14;

	mul0  = w0  * b0;
	mul1  = w1  * b1;
	mul2  = w2  * b2;
	mul3  = w3  * b3;
	mul4  = w4  * b4;
	mul5  = w5  * b5;
	mul6  = w6  * b6;
	mul7  = w7  * b7;
	mul8  = w8  * b8;
	mul9  = w9  * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;


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



/*void outer()
{

FIX_FM data[16][BUF_DPTH][22][42];
FIX_WT weights[BUF_DPTH][BUF_DPTH];

#pragma HLS array_partition variable=data dim=1 complete

CONV_1x1(data[0], data[1], weights);
CONV_1x1(data[0], data[2], weights);
CONV_1x1(data[0], data[3], weights);
CONV_1x1(data[0], data[4], weights);
CONV_1x1(data[0], data[5], weights);
CONV_1x1(data[0], data[6], weights);
CONV_1x1(data[0], data[7], weights);
CONV_1x1(data[0], data[8], weights);
CONV_1x1(data[0], data[9], weights);
CONV_1x1(data[0], data[10], weights);
CONV_1x1(data[0], data[11], weights);
CONV_1x1(data[0], data[12], weights);
CONV_1x1(data[0], data[13], weights);
CONV_1x1(data[0], data[14], weights);
CONV_1x1(data[0], data[15], weights);


}*/



void CONV_1x1(FIX_FM bottom[BUF_DPTH][22][42],
			  FIX_FM top[BUF_DPTH][22][42],
			  FIX_WT weights[BUF_DPTH][BUF_DPTH])
{
FIX_WT weight_buf[BUF_DPTH][BUF_DPTH];
FIX_32_12 tmp[8];

#pragma HLS array_partition variable=tmp dim=1 complete
//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete

#pragma HLS array_partition variable=top dim=1 cyclic factor=32
#pragma HLS array_partition variable=bottom dim=1 cyclic factor=32

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
//#pragma HLS pipeline
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




/*



FIX_32_12 compute_engine_32(  FIX_WT w0,  FIX_FM b0,
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
							  FIX_WT w15, FIX_FM b15,
							  FIX_WT w16, FIX_FM b16,
							  FIX_WT w17, FIX_FM b17,
							  FIX_WT w18, FIX_FM b18,
							  FIX_WT w19, FIX_FM b19,
							  FIX_WT w20, FIX_FM b20,
							  FIX_WT w21, FIX_FM b21,
							  FIX_WT w22, FIX_FM b22,
							  FIX_WT w23, FIX_FM b23,
							  FIX_WT w24, FIX_FM b24,
							  FIX_WT w25, FIX_FM b25,
							  FIX_WT w26, FIX_FM b26,
							  FIX_WT w27, FIX_FM b27,
							  FIX_WT w28, FIX_FM b28,
							  FIX_WT w29, FIX_FM b29,
							  FIX_WT w30, FIX_FM b30,
							  FIX_WT w31, FIX_FM b31)
{
//	#pragma HLS ALLOCATION instances=mul limit=8 operation

	FIX_32_12 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	FIX_32_12 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;

	FIX_32_12 mul16, mul17, mul18, mul19, mul20, mul21, mul22, mul23;
	FIX_32_12 mul24, mul25, mul26, mul27, mul28, mul29, mul30, mul31;

	FIX_32_12 add0, add1, add2,  add3,  add4,  add5,  add6,  add7;
	FIX_32_12 add8, add9, add10, add11, add12, add13, add14, add15, add16;

	FIX_32_12 add17, add18, add19, add20, add21, add22, add23, add24;
	FIX_32_12 add25, add26, add27, add28, add29, add30, add31, add32;

	mul0  = w0  * b0;
	mul1  = w1  * b1;
	mul2  = w2  * b2;
	mul3  = w3  * b3;
	mul4  = w4  * b4;
	mul5  = w5  * b5;
	mul6  = w6  * b6;
	mul7  = w7  * b7;
	mul8  = w8  * b8;
	mul9  = w9  * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;

	mul16  = w16  * b16;
	mul17  = w17  * b17;
	mul18  = w18  * b18;
	mul19  = w19  * b19;
	mul20  = w20  * b20;
	mul21  = w21  * b21;
	mul22  = w22  * b22;
	mul23  = w23  * b23;
	mul24  = w24  * b24;
	mul25  = w25  * b25;
	mul26  = w26  * b26;
	mul27  = w27  * b27;
	mul28  = w28  * b28;
	mul29  = w29  * b29;
	mul30  = w30  * b30;
	mul31  = w31  * b31;


	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;
	add4 = mul8  + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8  = mul16  + mul17;
	add9  = mul18  + mul19;
	add10 = mul20  + mul21;
	add11 = mul22  + mul23;
	add12 = mul24  + mul25;
	add13 = mul26  + mul27;
	add14 = mul28  + mul29;
	add15 = mul30  + mul31;

	add16  = add0 + add1;
	add17  = add2 + add3;
	add18  = add4 + add5;
	add19  = add6 + add7;

	add20  = add8  + add9;
	add21  = add10 + add11;
	add22  = add12 + add13;
	add23  = add14 + add15;

	add24 = add16  + add17;
	add25 = add18  + add19;

	add26 = add20 + add21;
	add27 = add22 + add23;

	add28 = add24 + add25;
	add29 = add26 + add27;

	add30 = add28 + add29;

	return add30;

}







void CONV_1x1(FIX_FM bottom[BUF_DPTH][22][42],
			  FIX_FM top[BUF_DPTH][22][42],
			  FIX_WT weights[BUF_DPTH][BUF_DPTH])
{
FIX_WT weight_buf[BUF_DPTH][BUF_DPTH];
FIX_32_12 tmp[8];

#pragma HLS array_partition variable=tmp dim=1 complete
#pragma HLS array_partition variable=top dim=1 block factor=32
#pragma HLS array_partition variable=bottom dim=1 block factor=32
#pragma HLS array_partition variable=weight_buf dim=1 complete
#pragma HLS array_partition variable=weight_buf dim=2 complete


	for(int i = 0; i < BUF_DPTH; i++)
		for(int j = 0; j < BUF_DPTH; j++)
			weight_buf[i][j] = weights[i][j];


	for(int h = 1; h <= 20; h++){
		for(int w = 1; w <= 40; w++) {

			for(int co = 0; co < BUF_DPTH; co+=8) {
#pragma HLS pipeline II=2

				for(int coo = 0; coo < 8; coo++) {
#pragma HLS unroll
					top[co+coo][h][w] += compute_engine_32(
											 weight_buf[co+coo][0],   bottom[0][h][w],
											 weight_buf[co+coo][1],   bottom[1][h][w],
											 weight_buf[co+coo][2],   bottom[2][h][w],
											 weight_buf[co+coo][3],   bottom[3][h][w],
											 weight_buf[co+coo][4],   bottom[4][h][w],
											 weight_buf[co+coo][5],   bottom[5][h][w],
											 weight_buf[co+coo][6],   bottom[6][h][w],
											 weight_buf[co+coo][7],   bottom[7][h][w],
											 weight_buf[co+coo][8],   bottom[8][h][w],
											 weight_buf[co+coo][9],   bottom[9][h][w],
											 weight_buf[co+coo][10],  bottom[10][h][w],
											 weight_buf[co+coo][11],  bottom[11][h][w],
											 weight_buf[co+coo][12],  bottom[12][h][w],
											 weight_buf[co+coo][13],  bottom[13][h][w],
											 weight_buf[co+coo][14],  bottom[14][h][w],
											 weight_buf[co+coo][15],  bottom[15][h][w],
											 weight_buf[co+coo][16],  bottom[16][h][w],
											 weight_buf[co+coo][17],  bottom[17][h][w],
											 weight_buf[co+coo][18],  bottom[18][h][w],
											 weight_buf[co+coo][19],  bottom[19][h][w],
											 weight_buf[co+coo][20],  bottom[20][h][w],
											 weight_buf[co+coo][21],  bottom[21][h][w],
											 weight_buf[co+coo][22],  bottom[22][h][w],
											 weight_buf[co+coo][23],  bottom[23][h][w],
											 weight_buf[co+coo][24],  bottom[24][h][w],
											 weight_buf[co+coo][25],  bottom[25][h][w],
											 weight_buf[co+coo][26],  bottom[26][h][w],
											 weight_buf[co+coo][27],  bottom[27][h][w],
											 weight_buf[co+coo][28],  bottom[28][h][w],
											 weight_buf[co+coo][29],  bottom[29][h][w],
											 weight_buf[co+coo][30],  bottom[30][h][w],
											 weight_buf[co+coo][31],  bottom[31][h][w]);
				}
			}
		}
	}
}


*/



