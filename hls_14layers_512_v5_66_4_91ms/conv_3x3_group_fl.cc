

// conv 3x3 for group (depth-wise convolutions)

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "net_hls.h"




void CONV_3x3_group(FIX_FM bottom[BUF_DPTH][22][42],
					FIX_FM top[BUF_DPTH][22][42],
					FIX_WT weights[BUF_DPTH][3][3],
					FIX_WT bias[BUF_DPTH][1],
					int skip)
{
	//FIX_WT weight_buf[BUF_DPTH];


	FIX_16_10 tmp[BUF_DPTH][22][42];
	//FIX_32_16 tmp[BUF_DPTH][22][42];


	for(int j = 0; j < 22; j++) {
		for(int k = 0; k < 42; k++) {
#pragma HLS pipeline
			for(int i = 0; i < 16; i++) {
				tmp[i][j][k] = bias[i][0];
			}
		}
	}

	for(int j = 0; j < 22; j++) {
		for(int k = 0; k < 42; k++) {
#pragma HLS pipeline
			for(int i = 16; i < 32; i++) {
				tmp[i][j][k] = bias[i][0];
			}
		}
	}


#pragma HLS array_partition variable=top dim=1 cyclic factor=16
#pragma HLS array_partition variable=bottom dim=1 cyclic factor=16
#pragma HLS array_partition variable=weights dim=1 cyclic factor=16
#pragma HLS array_partition variable=tmp dim=1 cyclic factor=16


	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){

			for(int h = 1; h <= 20; h++){
				for(int w = 1; w <= 40; w++){
#pragma HLS pipeline
					for(int co = 0; co < 16; co++){
#pragma HLS unroll

						//top[co][h][w] += weights[co][i][j] * (FIX_16_3)bottom[co][h+i-1][w+j-1];
						//top[co][h][w] += weights[co][i][j] * bottom[co][h+i-1][w+j-1];
						tmp[co][h][w] += weights[co][i][j] * (FIX_16_3)bottom[co][h+i-1][w+j-1];
					}
				}
			}
		}
	}


	if( skip == 0 ) {
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){

				for(int h = 1; h <= 20; h++){
					for(int w = 1; w <= 40; w++){
	#pragma HLS pipeline
						for(int co = 16; co < 32; co++){
	#pragma HLS unroll
							//top[co][h][w] += weights[co][i][j] * (FIX_16_3)bottom[co][h+i-1][w+j-1];
							//top[co][h][w] += weights[co][i][j] * bottom[co][h+i-1][w+j-1];
							tmp[co][h][w] += weights[co][i][j] * (FIX_16_3)bottom[co][h+i-1][w+j-1];
						}
					}
				}
			}
		}
	}

	for(int j = 0; j < 22; j++) {
		for(int k = 0; k < 42; k++) {
#pragma HLS pipeline
			for(int i = 0; i < 16; i++) {

				if( tmp[i][j][k] < 0)
					top[i][j][k] = 0;
				else if( tmp[i][j][k] > 4)
					top[i][j][k] = 4;
				else
					top[i][j][k] = tmp[i][j][k];
			}
		}
	}

	for(int j = 0; j < 22; j++) {
		for(int k = 0; k < 42; k++) {
#pragma HLS pipeline
			for(int i = 16; i < 32; i++) {
				if( tmp[i][j][k] < 0)
					top[i][j][k] = 0;
				else if( tmp[i][j][k] > 4)
					top[i][j][k] = 4;
				else
					top[i][j][k] = tmp[i][j][k];
			}
		}
	}
}

