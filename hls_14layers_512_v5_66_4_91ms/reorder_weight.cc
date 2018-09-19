


#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include "net_hls.h"


using namespace std;

/* float real input parameters */

extern float conv_1_weight_tmp[3][3][3];
extern float conv_1_bias_tmp[3];
extern float conv_2_weight_tmp[48][3];
extern float conv_2_bias_in[48];
extern float conv_4_weight_in[48][3][3];
extern float conv_4_bias_in[48];
extern float conv_5_weight_in[96][48];
extern float conv_5_bias_in[96];
extern float conv_7_weight_in[96][3][3];
extern float conv_7_bias_in[96];
extern float conv_8_weight_in[192][96];
extern float conv_8_bias_in[192];
extern float conv_10_weight_in[192][3][3];
extern float conv_10_bias_in[192];
extern float conv_11_weight_in[384][192];
extern float conv_11_bias_in[384];
extern float conv_12_weight_in[384][3][3];
extern float conv_12_bias_in[384];
extern float conv_13_weight_in[512][384];
extern float conv_13_bias_in[512];
extern float conv_14_weight_tmp[10][512];


/* fixed point parameters */

// cd : channel depth
// rd : reordered weight array depth
// bd : bias array depth


FIX_WT fix_conv_1_weight_in[BUF_DPTH][3][3];
FIX_WT fix_conv_1_bias_in[BUF_DPTH];

FIX_WT fix_conv_2_weight_in[48][BUF_DPTH];
FIX_WT fix_conv_2_weight_reorder[l2_rd][BUF_DPTH][BUF_DPTH];
FIX_WT fix_conv_2_bias_in[48];

FIX_WT fix_conv_4_weight_in[48][3][3];
FIX_WT fix_conv_4_bias_in[48];

FIX_WT fix_conv_5_weight_in[96][48];
FIX_WT fix_conv_5_weight_reorder[l5_rd][BUF_DPTH][BUF_DPTH];
FIX_WT fix_conv_5_bias_in[96];

FIX_WT fix_conv_7_weight_in[96][3][3];
FIX_WT fix_conv_7_bias_in[96];

FIX_WT fix_conv_8_weight_in[192][96];
FIX_WT fix_conv_8_weight_reorder[l8_rd][BUF_DPTH][BUF_DPTH];
FIX_WT fix_conv_8_bias_in[192];

FIX_WT fix_conv_10_weight_in[192][3][3];
FIX_WT fix_conv_10_bias_in[192];

FIX_WT fix_conv_11_weight_in[384][192];
FIX_WT fix_conv_11_weight_reorder[l11_rd][BUF_DPTH][BUF_DPTH];
FIX_WT fix_conv_11_bias_in[384];

FIX_WT fix_conv_12_weight_in[384][3][3];
FIX_WT fix_conv_12_bias_in[384];

FIX_WT fix_conv_13_weight_in[512][384];
FIX_WT fix_conv_13_weight_reorder[l13_rd][BUF_DPTH][BUF_DPTH];
FIX_WT fix_conv_13_bias_in[512];

FIX_WT fix_conv_14_weight_in[BUF_DPTH][512];
FIX_WT fix_conv_14_weight_reorder[l14_rd][BUF_DPTH][BUF_DPTH];


FIX_WT fix_conv_weight_1x1_all[all_1x1][BUF_DPTH][BUF_DPTH];
FIX_WT fix_conv_weight_3x3_all[all_3x3][BUF_DPTH][3][3];
FIX_WT fix_bias_all[all_bias][BUF_DPTH];
extern uint128 fix_conv_weight_1x1_all_128bit[all_1x1][BUF_DPTH/8][BUF_DPTH];
extern uint128 fix_conv_weight_3x3_all_128bit[all_3x3][BUF_DPTH/8][3][3];
extern uint128 fix_bias_all_128bit[all_bias][BUF_DPTH/8];

// correct index for BUF_DPTH=16
//index_1x1: 1180
//index_3x3: 45
//index_bias: 122



void reorder_weight_fix()
{

    std::ofstream ofs_param_write("params_512_fix.bin", std::ios::out | std::ios::binary);

    //for conv1
    for(int j = 0; j < 3; j++) {
    	for(int k = 0; k < 3; k++) {
    		for(int i = 0; i < BUF_DPTH; i++) {
    			if(i < 3) {
    				//for fixed-point data
    				fix_conv_1_weight_in[i][j][k] = (FIX_WT)conv_1_weight_tmp[i][j][k];
    				fix_conv_1_bias_in[i] = (FIX_WT)conv_1_bias_tmp[i];
    			}
    			else {
    				fix_conv_1_weight_in[i][j][k] = 0;
    				fix_conv_1_bias_in[i] = 0;
    			}
    		}
    	}
    }

    //for conv2
    for(int i = 0; i < 48; i++) {
    	fix_conv_2_bias_in[i] = conv_2_bias_in[i];
    	for(int j = 0; j < BUF_DPTH; j++) {
    		if(j < 3) {
    			fix_conv_2_weight_in[i][j] = (FIX_WT)conv_2_weight_tmp[i][j];
    		}
    		else {
    			fix_conv_2_weight_in[i][j] = 0;
    		}
    	}
    }

    // reorder conv2
    for(int col = 0; col < l2_cd; col++) {
    	for(int row = 0; row < l1_cd; row++) {
    		for(int i = 0; i < BUF_DPTH; i++) {
    			for(int j = 0; j < BUF_DPTH; j++) {
    				if( i + BUF_DPTH * col < 48 ) {
    					fix_conv_2_weight_reorder[col * l1_cd + row][i][j] = fix_conv_2_weight_in[i + BUF_DPTH * col][j + BUF_DPTH * row];
    				}
    				else {
    					fix_conv_2_weight_reorder[col * l1_cd + row][i][j] = 0;
    				}
    			}
    		}
    	}
    }

    //for conv4
    for(int j = 0; j < 48; j++) {
    	fix_conv_4_bias_in[j] = (FIX_WT)conv_4_bias_in[j];
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_4_weight_in[j][k][i] = (FIX_WT)conv_4_weight_in[j][k][i];
			}
		}
	}


    //for conv5
    for(int j = 0; j < 96; j++) {
		fix_conv_5_bias_in[j] = (FIX_WT)conv_5_bias_in[j];
		for(int k = 0; k < 48; k++) {
			fix_conv_5_weight_in[j][k] = (FIX_WT)conv_5_weight_in[j][k];

		}
	}

    // reorder conv5
    for(int col = 0; col < l5_cd; col++) {
    	for(int row = 0; row < l4_cd; row++) {
    		for(int i = 0; i < BUF_DPTH; i++) {
    			for(int j = 0; j < BUF_DPTH; j++) {
    				fix_conv_5_weight_reorder[col * l4_cd + row][i][j] = fix_conv_5_weight_in[i + col * BUF_DPTH][j + row * BUF_DPTH];
    			}
    		}
    	}
    }


    //for conv7
	for(int j = 0; j < 96; j++) {
		fix_conv_7_bias_in[j] = (FIX_WT)conv_7_bias_in[j];
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_7_weight_in[j][k][i] = (FIX_WT)conv_7_weight_in[j][k][i];
			}
		}
	}

    //for conv8
	for(int i = 0; i < 192; i++) {
		fix_conv_8_bias_in[i] = (FIX_WT)conv_8_bias_in[i];
		for(int j = 0; j < 96; j++) {
			fix_conv_8_weight_in[i][j] = (FIX_WT)conv_8_weight_in[i][j];
		}
	}

	//reorder conv8
	for(int col = 0; col < l8_cd; col++ ) {
		for(int row = 0; row < l7_cd; row++) {
			for(int i = 0; i < BUF_DPTH; i++) {
				for(int j = 0; j < BUF_DPTH; j++) {
					fix_conv_8_weight_reorder[col * l7_cd + row][i][j] = fix_conv_8_weight_in[i + col * BUF_DPTH][j + row * BUF_DPTH];
				}
			}
		}
	}


	//for conv10
	for(int j = 0; j < 192; j++) {
		fix_conv_10_bias_in[j] = (FIX_WT)conv_10_bias_in[j];
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_10_weight_in[j][k][i] = (FIX_WT)conv_10_weight_in[j][k][i];
			}
		}
	}

	//for conv11
	for(int i = 0; i < 384; i++) {
		fix_conv_11_bias_in[i] = (FIX_WT)conv_11_bias_in[i];
		for(int j = 0; j < 192; j++) {
			fix_conv_11_weight_in[i][j] = (FIX_WT)conv_11_weight_in[i][j];
		}
	}

	//// reorder conv_11_weight
	for(int col = 0; col < l11_cd; col++ ) {
		for(int row = 0; row < l10_cd; row++) {
			for(int i = 0; i < BUF_DPTH; i++) {
				for(int j = 0; j < BUF_DPTH; j++) {
					fix_conv_11_weight_reorder[col * l10_cd + row][i][j] = fix_conv_11_weight_in[i + col * BUF_DPTH][j + row * BUF_DPTH];
				}
			}
		}
	}

	//for conv12
	for(int j = 0; j < 384; j++) {
		fix_conv_12_bias_in[j] = (FIX_WT)conv_12_bias_in[j];
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_12_weight_in[j][k][i] = (FIX_WT)conv_12_weight_in[j][k][i];
			}
		}
	}


	//for conv13
	for(int i = 0; i < 512; i++) {
		fix_conv_13_bias_in[i] = (FIX_WT)conv_13_bias_in[i];
		for(int j = 0; j < 384; j++) {
			fix_conv_13_weight_in[i][j] = (FIX_WT)conv_13_weight_in[i][j];
		}
	}

	//// reorder conv_13_weight
	for(int col = 0; col < l13_cd; col++ ) {
		for(int row = 0; row < l12_cd; row++) {
			for(int i = 0; i < BUF_DPTH; i++) {
				for(int j = 0; j < BUF_DPTH; j++) {
					fix_conv_13_weight_reorder[col * l12_cd + row][i][j] = fix_conv_13_weight_in[i + col * BUF_DPTH][j + row * BUF_DPTH];
				}
			}
		}
	}


	//for conv14
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 512; j++) {
			if(i < 10) {
				fix_conv_14_weight_in[i][j] = (FIX_WT)conv_14_weight_tmp[i][j];
			}
			else{
				fix_conv_14_weight_in[i][j] = (FIX_WT)0.0;
			}
		}
	}

	//// reorder conv_14_weight
	for(int col = 0; col < l14_cd; col++ ) {
		for(int row = 0; row < l13_cd; row++) {
			for(int i = 0; i < BUF_DPTH; i++) {
				for(int j = 0; j < BUF_DPTH; j++) {
					fix_conv_14_weight_reorder[col * l13_cd + row][i][j] = fix_conv_14_weight_in[i + col * BUF_DPTH][j + row * BUF_DPTH];
				}
			}
		}
	}


	//////////// put all reordered weights together

	// copy conv_1 to conv_weight_3x3_all
	int index_3x3 = 0;
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++)
				fix_conv_weight_3x3_all[index_3x3][i][j][k] = fix_conv_1_weight_in[i][j][k];
		}
	}

	// copy conv_4 to conv_weight_3x3_all
	for(int i = 0; i < l4_rd * BUF_DPTH; i++) {
		if( i % BUF_DPTH == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				if( i < 48 ) {
					fix_conv_weight_3x3_all[index_3x3][i%BUF_DPTH][j][k] = fix_conv_4_weight_in[i][j][k];
				}
				else{
					fix_conv_weight_3x3_all[index_3x3][i%BUF_DPTH][j][k] = 0;
				}
			}
		}
	}


	// copy conv_7 to conv_weight_3x3_all
	for(int i = 0; i < 96; i++) {
		if( i % BUF_DPTH == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%BUF_DPTH][j][k] = fix_conv_7_weight_in[i][j][k];
			}
		}
	}

	// copy conv_10 to conv_weight_3x3_all
	for(int i = 0; i < 192; i++) {
		if( i % BUF_DPTH == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%BUF_DPTH][j][k] = fix_conv_10_weight_in[i][j][k];
			}
		}
	}

	// copy conv_12 to conv_weight_3x3_all
	for(int i = 0; i < 384; i++) {
		if( i % BUF_DPTH == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%BUF_DPTH][j][k] = fix_conv_12_weight_in[i][j][k];
			}
		}
	}

	// copy conv_2_reorder to conv_weight_1x1_all
	int index_1x1 = -1;
	for(int i = 0; i < l2_rd; i++) {
		index_1x1++;

		for(int j = 0; j < BUF_DPTH; j++) {
			for(int k = 0; k < BUF_DPTH; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_2_weight_reorder[i][j][k];
			}
		}
	}



	// copy conv_5_reorder to conv_weight_1x1_all
	for(int i = 0; i < l5_rd; i++) {
		index_1x1++;

		for(int j = 0; j < BUF_DPTH; j++) {
			for(int k = 0; k < BUF_DPTH; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_5_weight_reorder[i][j][k];
			}
		}
	}


	// copy conv_8_reorder to conv_weight_1x1_all
	for(int i = 0; i < l8_rd; i++) {
		index_1x1++;

		for(int j = 0; j < BUF_DPTH; j++) {
			for(int k = 0; k < BUF_DPTH; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_8_weight_reorder[i][j][k];
			}
		}
	}

	// copy conv_11_reorder to conv_weight_1x1_all
	for(int i = 0; i < l11_rd; i++) {
		index_1x1++;

		for(int j = 0; j < BUF_DPTH; j++) {
			for(int k = 0; k < BUF_DPTH; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_11_weight_reorder[i][j][k];
			}
		}
	}



	// copy conv_13_reorder to conv_weight_1x1_all
	for(int i = 0; i < l13_rd; i++) {
		index_1x1++;

		for(int j = 0; j < BUF_DPTH; j++) {
			for(int k = 0; k < BUF_DPTH; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_13_weight_reorder[i][j][k];
			}
		}
	}




	// copy conv_14_reorder to conv_weight_1x1_all
	for(int i = 0; i < l14_rd; i++) {
		index_1x1++;

		for(int j = 0; j < BUF_DPTH; j++) {
			for(int k = 0; k < BUF_DPTH; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_14_weight_reorder[i][j][k];
			}
		}
	}


	// put all bias into one array fix_bias_all[93][16][16]
	// copy conv_1_bias to fix_bias_all
	int index_bias = 0;
	for(int i = 0; i < BUF_DPTH; i++) {
		fix_bias_all[index_bias][i] = fix_conv_1_bias_in[i];
	}

	// copy conv_2_bias to fix_bias_all
	for(int ch = 0; ch < l2_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {

			if( i + ch * BUF_DPTH < 48 ) {
				fix_bias_all[index_bias][i] = fix_conv_2_bias_in[ch * BUF_DPTH + i];
			}
			else {
				fix_bias_all[index_bias][i] = 0;
			}
		}
	}


	// copy conv_4_bias to fix_bias_all
	for(int ch = 0; ch < l4_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {

			if( ch * BUF_DPTH + i < 48) {
				fix_bias_all[index_bias][i] = fix_conv_4_bias_in[ch * BUF_DPTH + i];
			}
			else {
				fix_bias_all[index_bias][i] = 0;
			}
		}
	}






	// copy conv_5_bias to fix_bias_all
	for(int ch = 0; ch < l5_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {
			fix_bias_all[index_bias][i] = fix_conv_5_bias_in[ch * BUF_DPTH + i];
		}
	}


	// copy conv_7_bias to fix_bias_all
	for(int ch = 0; ch < l7_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {
			fix_bias_all[index_bias][i] = fix_conv_7_bias_in[ch * BUF_DPTH + i];
		}
	}

	// copy conv_8_bias to fix_bias_all
	for(int ch = 0; ch < l8_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {
			fix_bias_all[index_bias][i] = fix_conv_8_bias_in[ch * BUF_DPTH + i];
		}
	}


	// copy conv_10_bias to fix_bias_all
	for(int ch = 0; ch < l10_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {
			fix_bias_all[index_bias][i] = fix_conv_10_bias_in[ch * BUF_DPTH + i];
		}
	}

	// copy conv_11_bias to fix_bias_all
	for(int ch = 0; ch < l11_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {
			fix_bias_all[index_bias][i] = fix_conv_11_bias_in[ch * BUF_DPTH + i];
		}
	}


	// copy conv_12_bias to fix_bias_all
	for(int ch = 0; ch < l12_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {
			fix_bias_all[index_bias][i] = fix_conv_12_bias_in[ch * BUF_DPTH + i];
		}
	}


	// copy conv_13_bias to fix_bias_all
	for(int ch = 0; ch < l13_bd; ch++) {
		index_bias++;

		for(int i = 0; i < BUF_DPTH; i++) {
			fix_bias_all[index_bias][i] = fix_conv_13_bias_in[ch * BUF_DPTH + i];
		}
	}

	printf("index_1x1: %d\n", index_1x1);
	printf("index_3x3: %d\n", index_3x3);
	printf("index_bias: %d\n", index_bias);


	printf("all_1x1: %d\n", all_1x1);
	printf("all_3x3: %d\n", all_3x3);
	printf("all_bias: %d\n", all_bias);


    // write conv_1x1 weights into params_fix_384.bin
    //ofs_param_write.write((char*)fix_conv_weight_1x1_all, all_1x1*BUF_DPTH*BUF_DPTH*sizeof(FIX_WT));
	//ofs_param_write.write((char*)fix_conv_weight_1x1_all, all_1x1*32*32*sizeof(FIX_WT));

	//ofs_param_write.write((char*)fix_conv_weight_1x1_all, (index_1x1+1)*32*32*sizeof(FIX_16_5));

	// write conv_3x3 into params_fix_384.bin
    //ofs_param_write.write((char*)fix_conv_weight_3x3_all, (index_3x3+1)*BUF_DPTH*3*3*sizeof(FIX_16_5));

    // write bias_all into params_fix_384.bin
    //ofs_param_write.write((char*)fix_bias_all, (index_bias+1)*BUF_DPTH*sizeof(FIX_16_5));

    //ofs_param_write.close();



    // fill fix_conv_weight_1x1_all into 128bit width bus
    for(int i = 0; i < all_1x1; i++) {
    	for(int k = 0; k < BUF_DPTH; k++) {

    		for(int j = 0; j < BUF_DPTH; j += 8) {
    			uint128 DATA = 0;

    			DATA.range(9, 0) = fix_conv_weight_1x1_all[i][j][k].range(9, 0);
    			DATA.range(16+8, 16-1) = fix_conv_weight_1x1_all[i][j+1][k].range(9, 0);
    			DATA.range(16*2+8, 16*2-1) = fix_conv_weight_1x1_all[i][j+2][k].range(9, 0);
    			DATA.range(16*3+8, 16*3-1) = fix_conv_weight_1x1_all[i][j+3][k].range(9, 0);
    			DATA.range(16*4+8, 16*4-1) = fix_conv_weight_1x1_all[i][j+4][k].range(9, 0);
    			DATA.range(16*5+8, 16*5-1) = fix_conv_weight_1x1_all[i][j+5][k].range(9, 0);
    			DATA.range(16*6+8, 16*6-1) = fix_conv_weight_1x1_all[i][j+6][k].range(9, 0);
    			DATA.range(16*7+8, 16*7-1) = fix_conv_weight_1x1_all[i][j+7][k].range(9, 0);

    			fix_conv_weight_1x1_all_128bit[i][j/8][k].range(127, 0) = DATA.range(127, 0);
    		}
    	}
    }

    ofs_param_write.write((char*)fix_conv_weight_1x1_all_128bit, 306 * 4 * 32 * sizeof(uint128));



    // fill fix_conv_weight_3x3_all into 128bit width bus
    for(int i = 0; i < all_3x3; i++) {
    	for(int m = 0; m < 3; m++) {
    		for(int n = 0; n < 3; n++) {

				for(int j = 0; j < BUF_DPTH; j += 8) {
					uint128 DATA = 0;

					DATA.range(9, 0) = fix_conv_weight_3x3_all[i][j][m][n].range(9, 0);
					DATA.range(16+8, 16-1) = fix_conv_weight_3x3_all[i][j+1][m][n].range(9, 0);
					DATA.range(16*2+8, 16*2-1) = fix_conv_weight_3x3_all[i][j+2][m][n].range(9, 0);
					DATA.range(16*3+8, 16*3-1) = fix_conv_weight_3x3_all[i][j+3][m][n].range(9, 0);
					DATA.range(16*4+8, 16*4-1) = fix_conv_weight_3x3_all[i][j+4][m][n].range(9, 0);
					DATA.range(16*5+8, 16*5-1) = fix_conv_weight_3x3_all[i][j+5][m][n].range(9, 0);
					DATA.range(16*6+8, 16*6-1) = fix_conv_weight_3x3_all[i][j+6][m][n].range(9, 0);
					DATA.range(16*7+8, 16*7-1) = fix_conv_weight_3x3_all[i][j+7][m][n].range(9, 0);

					fix_conv_weight_3x3_all_128bit[i][j/8][m][n].range(127, 0) = DATA.range(127, 0);



				}
    		}
    	}
    }
    ofs_param_write.write((char*)fix_conv_weight_3x3_all_128bit, 24 * 32 / 8 * 3 * 3 * sizeof(uint128));


    // fill fix_bias_all into 128bit width bus
    for(int i = 0; i < all_bias; i++) {
		for(int j = 0; j < BUF_DPTH; j+=8) {
			uint128 DATA = 0;

			DATA.range(9, 0) = fix_bias_all[i][j].range(9, 0);
			DATA.range(16+8, 16-1) = fix_bias_all[i][j+1].range(9, 0);
			DATA.range(16*2+8, 16*2-1) = fix_bias_all[i][j+2].range(9, 0);
			DATA.range(16*3+8, 16*3-1) = fix_bias_all[i][j+3].range(9, 0);
			DATA.range(16*4+8, 16*4-1) = fix_bias_all[i][j+4].range(9, 0);
			DATA.range(16*5+8, 16*5-1) = fix_bias_all[i][j+5].range(9, 0);
			DATA.range(16*6+8, 16*6-1) = fix_bias_all[i][j+6].range(9, 0);
			DATA.range(16*7+8, 16*7-1) = fix_bias_all[i][j+7].range(9, 0);

			fix_bias_all_128bit[i][j/8].range(127, 0) = DATA.range(127, 0);
		}
    }
    ofs_param_write.write((char*)fix_bias_all_128bit, 63 * 32 / 8 * sizeof(uint128));
    ofs_param_write.close();


    printf("1x1 [0][0][0]: %s\n", fix_conv_weight_1x1_all_128bit[0][0][0].to_string().c_str());
    printf("1x1 [0][0][0]: %s\n", fix_conv_weight_1x1_all_128bit[1][0][0].to_string().c_str());

}

