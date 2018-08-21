

// conv 3x3 for group (depth-wise convolutions)

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "net_hls.h"


void load_weights(FIX_WT weight_buf[BUF_DPTH],
				  FIX_WT weights[BUF_DPTH][3][3],
				  int i, int j)
{
#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=weight_buf dim=1 complete

	for(int coo = 0; coo < BUF_DPTH; coo++){
#pragma HLS unroll
		weight_buf[coo] = weights[coo][i][j];
		//printf("weight_buf[%d] = weights[%d][%d][%d] (%f)\n", coo, coo, i, j, weights[coo][i][j] );
	}
}


void CONV_3x3_group(FIX_FM bottom[BUF_DPTH][22][42],
					FIX_FM top[BUF_DPTH][22][42],
					FIX_WT weights[BUF_DPTH][3][3])
{
	//FIX_WT weight_buf[BUF_DPTH];

#pragma HLS array_partition variable=top dim=1 cyclic factor=32
#pragma HLS array_partition variable=bottom dim=1 cyclic factor=32
#pragma HLS array_partition variable=weights dim=1 cyclic factor=32

//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete



	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){

//#pragma HLS dataflow

			//load_weights(weight_buf, weights, i, j);

			for(int h = 1; h <= 20; h++){
				for(int w = 1; w <= 40; w++){
#pragma HLS pipeline
					for(int co = 0; co < BUF_DPTH; co++){
#pragma HLS unroll
						//top_tmp[co][h][w] += weight_buf[co] * bottom[co][h+i-1][w+j-1];
						//top[co][h][w] += weight_buf[co] * bottom[co][h+i-1][w+j-1];
						top[co][h][w] += weights[co][i][j] * bottom[co][h+i-1][w+j-1];
					}
				}
			}
		}
	}

}

