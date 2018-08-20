#include "net_hls.h"
#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>
#include <string.h>


using namespace std;



// feature map buffers
FIX_FM FM_bufs[16][BUF_DPTH][22][42];
FIX_FM FM_buf_pool[BUF_DPTH][10][20];

FIX_WT weight_buf_1x1_1[BUF_DPTH][BUF_DPTH];
FIX_WT weight_buf_1x1_2[BUF_DPTH][BUF_DPTH];
FIX_WT weight_buf_1x1_3[BUF_DPTH][BUF_DPTH];
FIX_WT weight_buf_1x1_4[BUF_DPTH][BUF_DPTH];

FIX_WT weight_buf_3x3_1[BUF_DPTH][3][3];
FIX_WT weight_buf_3x3_2[BUF_DPTH][3][3];
FIX_WT weight_buf_3x3_3[BUF_DPTH][3][3];
FIX_WT weight_buf_3x3_4[BUF_DPTH][3][3];

FIX_WT bias_buf[BUF_DPTH][BUF_DPTH];


void fill_output( int layer, float buf[BUF_DPTH][22][42], int ch, int col, int row);
void fill_output_pool( int layer, float buf[BUF_DPTH][10][20], int ch, int col, int row);

void fill_output_fix( int layer, FIX_FM buf[BUF_DPTH][22][42], int ch, int col, int row);
void fill_output_last_fix( int layer, FIX_FM_last buf[10][22][42], int ch, int col, int row);
void fill_output_pool_fix( int layer, FIX_FM buf[BUF_DPTH][10][20], int ch, int col, int row);
void output_PL_layers();

int PL_golden_compare_layer_1();
int PL_golden_compare_layer_2();
int PL_golden_compare_layer_3();
int PL_golden_compare_layer_4();
int PL_golden_compare_layer_5();
int PL_golden_compare_layer_6();
int PL_golden_compare_layer_7();
int PL_golden_compare_layer_8();
int PL_golden_compare_layer_9();
int PL_golden_compare_layer_10();
int PL_golden_compare_layer_11();
int PL_golden_compare_layer_12();
int PL_golden_compare_layer_13();
int PL_golden_compare_layer_14();


FIX_32_25 my_exp_fix(FIX_FM input)
{
#pragma HLS latency min=2 max=20
	FIX_32_25 output;

	output = (FIX_32_25)exp((float)input);

	//printf("input: %f, output: %f\n", input.to_float(), output.to_float());

	return output;
}


void compute_bounding_box(float predict_box[5])
{
    int batch_size = 1;
    int num_anchors = 2;
    int h = 20;
    int w = 40;

    FIX_32_4 box[4] = {1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669};

    FIX_32_4 conf_thresh = 0.0;
    int conf_j = 0;
    int conf_m = 0;
    int conf_n = 0;

    FIX_32_4 conf_box1 = 0.0;
    FIX_32_4 conf_box2 = 0.0;

    for(int m = 1; m <= h; m++){
        for(int n = 1 ;n <= w; n++){
            conf_box1 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_bufs[15][4][m][n]));
            if(conf_box1 > conf_thresh){
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

            }
        }
    }

    for(int m = 1; m <= h; m++){
        for(int n = 1; n <= w; n++){
            conf_box2 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_bufs[15][9][m][n]));
            if(conf_box2 > conf_thresh){
                conf_thresh = conf_box2;
                conf_j = 1;
                conf_m = m;
                conf_n = n;
            }
        }
    }


    printf("\n\n");


    if( conf_j == 0 ) {
        // first bounding box
        predict_box[0] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_bufs[15][0][conf_m][conf_n])) + (FIX_32_25)(conf_n-1);
        predict_box[1] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_bufs[15][1][conf_m][conf_n])) + (FIX_32_25)(conf_m-1);
        predict_box[2] = my_exp_fix(FM_bufs[15][2][conf_m][conf_n]) * box[0];
        predict_box[3] = my_exp_fix(FM_bufs[15][3][conf_m][conf_n]) * box[1];
        predict_box[4] = conf_thresh;
    }
    else if( conf_j == 1 ) {
        // second bounding box
        predict_box[0] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_bufs[15][5][conf_m][conf_n])) + (FIX_32_25)(conf_n-1);
        predict_box[1] = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_bufs[15][6][conf_m][conf_n])) + (FIX_32_25)(conf_m-1);
        predict_box[2] = my_exp_fix(FM_bufs[15][7][conf_m][conf_n]) * box[2];
        predict_box[3] = my_exp_fix(FM_bufs[15][8][conf_m][conf_n]) * box[3];
        predict_box[4] = conf_thresh;
    }


    printf("\n\nPL output:\n");
    printf("conf_m: %d, conf_n:%d\n\n", conf_m-1, conf_n-1);

}


void buffer_copy_to_axi( FIX_FM dest[BUF_DPTH][22][42], FIX_FM src[BUF_DPTH][22][42])
{
	//memcpy(dest, src, sizeof(FIX_FM)*16*22*42);

	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 22; j++) {
			for(int k = 0; k < 42; k++) {
//#pragma HLS pipeline
				dest[i][j][k] = src[i][j][k];
			}
		}
	}
}

void buffer_copy_from_axi( FIX_FM dest[BUF_DPTH][22][42], FIX_FM src[BUF_DPTH][22][42])
{
	//memcpy(dest, src, sizeof(FIX_FM)*16*22*42);

	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 22; j++) {
			for(int k = 0; k < 42; k++) {
//#pragma HLS pipeline
				dest[i][j][k] = src[i][j][k];
			}
		}
	}

}


void load_weight_2D_from_axi( FIX_WT dest[BUF_DPTH][BUF_DPTH], FIX_WT src[BUF_DPTH][BUF_DPTH])
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < BUF_DPTH; j++) {
//#pragma HLS pipeline
			dest[i][j] = src[i][j];
		}
	}
}

void load_weight_3D_from_axi( FIX_WT dest[BUF_DPTH][3][3], FIX_WT src[BUF_DPTH][3][3])
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
//#pragma HLS pipeline
				dest[i][j][k] = src[i][j][k];
			}
		}
	}
}


void load_bias_from_axi(FIX_WT dest[BUF_DPTH][BUF_DPTH], FIX_WT src[BUF_DPTH])
{
	for(int i = 0; i < BUF_DPTH; i++) {
		dest[i][0] = src[i];
	}
}

void set_bias( FIX_FM buf[BUF_DPTH][22][42], FIX_WT bias[BUF_DPTH][BUF_DPTH])
{
	for(int j = 1; j <= 20; j++) {
		for(int k = 1; k <= 40; k++) {
//#pragma HLS pipeline

			for(int i = 0; i < BUF_DPTH; i++)
				buf[i][j][k] = bias[i][0];

		}
	}
}

void copy_to_DDR_pool9( FIX_FM dest[BUF_DPTH][22][42], FIX_FM buf[BUF_DPTH][10][20], int b_col, int b_row )
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 10; j++) {
			for(int k = 0; k < 20; k++) {
//#pragma HLS pipeline
				dest[i][j+1 + b_col*10][k+1 + b_row*20] = buf[i][j][k];
			}
		}
	}
}




void copy_to_DDR_pool3( FIX_FM ddr_pool3[l3_cd * BUF_DPTH][82][162], FIX_FM buf[BUF_DPTH][10][20], int ch, int col, int row)
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 10; j++) {
			for(int k = 0; k < 20; k++) {
//#pragma HLS pipeline
				ddr_pool3[i + ch*BUF_DPTH][j+1 + col*10][k+1 + row*20] = buf[i][j][k];
			}
		}
	}
}


void copy_to_DDR_pool6( FIX_FM ddr_pool6[l6_cd * BUF_DPTH][42][82], FIX_FM buf[BUF_DPTH][10][20], int ch, int col, int row)
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 10; j++) {
			for(int k = 0; k < 20; k++) {
//#pragma HLS pipeline
				ddr_pool6[i + ch*BUF_DPTH][j+1 + col*10][k+1 + row*20] = buf[i][j][k];
			}
		}
	}
}



void load_pool3_from_axi(FIX_FM buf[BUF_DPTH][22][42], FIX_FM DDR_pool3[48][82][162],
							int ch, int col, int row)
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int h = 0; h < 22; h++) {
			for(int w = 0; w < 42; w++ ) {
//#pragma HLS pipeline
				buf[i][h][w] = DDR_pool3[i + ch*BUF_DPTH][h + col*20][w + row*40];
			}
		}
	}
}


void load_pool6_from_axi(FIX_FM buf[BUF_DPTH][22][42], FIX_FM DDR_pool6[96][42][82],
							int ch, int col, int row)
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int h = 0; h < 22; h++) {
			for(int w = 0; w < 42; w++ ) {
//#pragma HLS pipeline
				buf[i][h][w] = DDR_pool6[i + ch*BUF_DPTH][h + col*20][w + row*40];
			}
		}
	}
}



FIX_FM img_norm_ch[256] = {
		-2.000000, -1.984314, -1.968627, -1.952941, -1.937255, -1.921569, -1.905882, -1.890196, -1.874510, -1.858824, -1.843137, -1.827451, -1.811765, -1.796078, -1.780392, -1.764706, -1.749020,
		-1.733333, -1.717647, -1.701961, -1.686275, -1.670588, -1.654902, -1.639216, -1.623529, -1.607843, -1.592157, -1.576471, -1.560784, -1.545098, -1.529412, -1.513725, -1.498039,
		-1.482353, -1.466667, -1.450980, -1.435294, -1.419608, -1.403922, -1.388235, -1.372549, -1.356863, -1.341176, -1.325490, -1.309804, -1.294118, -1.278431, -1.262745, -1.247059,
		-1.231373, -1.215686, -1.200000, -1.184314, -1.168627, -1.152941, -1.137255, -1.121569, -1.105882, -1.090196, -1.074510, -1.058824, -1.043137, -1.027451, -1.011765, -0.996078,
		-0.980392, -0.964706, -0.949020, -0.933333, -0.917647, -0.901961, -0.886275, -0.870588, -0.854902, -0.839216, -0.823529, -0.807843, -0.792157, -0.776471, -0.760784, -0.745098,
		-0.729412, -0.713725, -0.698039, -0.682353, -0.666667, -0.650980, -0.635294, -0.619608, -0.603922, -0.588235, -0.572549, -0.556863, -0.541176, -0.525490, -0.509804, -0.494118,
		-0.478431, -0.462745, -0.447059, -0.431373, -0.415686, -0.400000, -0.384314, -0.368627, -0.352941, -0.337255, -0.321569, -0.305882, -0.290196, -0.274510, -0.258824, -0.243137,
		-0.227451, -0.211765, -0.196078, -0.180392, -0.164706, -0.149020, -0.133333, -0.117647, -0.101961, -0.086275, -0.070588, -0.054902, -0.039216, -0.023529, -0.007843, 0.007843,
		0.023529, 0.039216, 0.054902, 0.070588, 0.086275, 0.101961, 0.117647, 0.133333, 0.149020, 0.164706, 0.180392, 0.196078, 0.211765, 0.227451, 0.243137, 0.258824,
		0.274510, 0.290196, 0.305882, 0.321569, 0.337255, 0.352941, 0.368627, 0.384314, 0.400000, 0.415686, 0.431373, 0.447059, 0.462745, 0.478431, 0.494118, 0.509804,
		0.525490, 0.541176, 0.556863, 0.572549, 0.588235, 0.603922, 0.619608, 0.635294, 0.650980, 0.666667, 0.682353, 0.698039, 0.713725, 0.729412, 0.745098, 0.760784,
		0.776471, 0.792157, 0.807843, 0.823529, 0.839216, 0.854902, 0.870588, 0.886275, 0.901961, 0.917647, 0.933333, 0.949020, 0.964706, 0.980392, 0.996078, 1.011765,
		1.027451, 1.043137, 1.058824, 1.074510, 1.090196, 1.105882, 1.121569, 1.137255, 1.152941, 1.168627, 1.184314, 1.200000, 1.215686, 1.231373, 1.247059, 1.262745,
		1.278431, 1.294118, 1.309804, 1.325490, 1.341176, 1.356863, 1.372549, 1.388235, 1.403922, 1.419608, 1.435294, 1.450980, 1.466667, 1.482353, 1.498039, 1.513725,
		1.529412, 1.545098, 1.560784, 1.576471, 1.592157, 1.607843, 1.623529, 1.639216, 1.654902, 1.670588, 1.686275, 1.701961, 1.717647, 1.733333, 1.749020, 1.764706,
		1.780392, 1.796078, 1.811765, 1.827451, 1.843137, 1.858824, 1.874510, 1.890196, 1.905882, 1.921569, 1.937255, 1.952941, 1.968627, 1.984314, 2.000000
};


void load_image_chunk_norm(FIX_FM img_buf[BUF_DPTH][22][42], uint8 image_in_raw_pad[3][162][322],
							int col, int row)
{
	for(int i = 0; i < 22; i++) {
		for(int j = 0; j < 42; j++) {
#ifdef CSIM_DEBUG
			if(i + col*20 == 0 || i + col*20 == 161 || j + row*40 == 0 || j + row*40 == 321 )
				img_buf[0][i][j] = 0.0;
			else
#endif
				img_buf[0][i][j] = img_norm_ch[(image_in_raw_pad[0][i + col*20][j + row*40]).to_uint()];
		}
	}

	for(int i = 0; i < 22; i++) {
		for(int j = 0; j < 42; j++) {
#ifdef CSIM_DEBUG
			if(i + col*20 == 0 || i + col*20 == 161 || j + row*40 == 0 || j + row*40 == 321 )
				img_buf[1][i][j] = 0.0;
			else
#endif
				img_buf[1][i][j] = img_norm_ch[(image_in_raw_pad[1][i + col*20][j + row*40]).to_uint()];
		}
	}

	for(int i = 0; i < 22; i++) {
		for(int j = 0; j < 42; j++) {
#ifdef CSIM_DEBUG
			if(i + col*20 == 0 || i + col*20 == 161 || j + row*40 == 0 || j + row*40 == 321 )
				img_buf[2][i][j] = 0.0;
			else
#endif
				img_buf[2][i][j] = img_norm_ch[(image_in_raw_pad[2][i + col*20][j + row*40]).to_uint()];
		}
	}
}




inline FIX_FM max(FIX_FM a, FIX_FM b, FIX_FM c, FIX_FM d)
{
	FIX_FM t1, t2;

	if(a > b) t1 = a;
	else t1 = b;

	if(c > d) t2 = c;
	else t2 = d;

	if(t1 > t2) return t1;
	else return t2;
}


void max_pooling(FIX_FM buf_in[BUF_DPTH][22][42], FIX_FM buf_out[BUF_DPTH][10][20])
{

	for(int i = 0; i < 10; i++) {
		for(int j = 0; j < 20; j++) {
//#pragma HLS pipeline
			for(int ch = 0; ch < BUF_DPTH; ch++) {
//#pragma HLS unroll
				buf_out[ch][i][j] = max(buf_in[ch][i*2+1][j*2+1], buf_in[ch][i*2+1][j*2+2],
								     	  buf_in[ch][i*2+2][j*2+1], buf_in[ch][i*2+2][j*2+2]);
			}
		}
	}
}


void clear_buf( FIX_FM buf[BUF_DPTH][22][42])
{

	for(int j = 0; j < 22; j++) {
		for(int k = 0; k < 42; k++) {
			for(int i = 0; i < BUF_DPTH; i++) {
				buf[i][j][k] = 0;
			}
		}
	}
}


void clear_padding( FIX_FM buf[BUF_DPTH][22][42])
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 0; j < 22; j++) {
				buf[i][j][0] = 0;
				buf[i][j][41] = 0;
		}
		for(int k = 0; k < 42; k++) {
				buf[i][0][k] = 0;
				buf[i][23][k] = 0;
		}
	}
}


void Relu( FIX_FM buf[BUF_DPTH][22][42] )
{
	for(int j = 1; j <= 20; j++) {
		for(int k = 1; k <= 40; k++) {
//#pragma HLS pipeline
			for(int i = 0; i < BUF_DPTH; i++) {
//#pragma HLS unroll
				if( buf[i][j][k] < 0 ) {
					buf[i][j][k] = 0;
				}
			}
		}
	}
}



// ch col row are offsets corresponding to feature map
void print_buf( float buf[BUF_DPTH][22][42], int ch, int col, int row)
{
	for(int i = 0; i < BUF_DPTH; i++) {
		for(int j = 1; j <= 20; j++) {
			for(int k = 1; k <= 40; k++) {
				printf("buf output[%d][%d][%d] = %f\n", ch*BUF_DPTH+i, col*20+j-1, row*40+k-1, buf[i][j][k]);
			}
		}
	}
}


void mobilenet(uint8 image_in_raw_pad[3][162][322],

				FIX_WT conv_weight_1x1_all[all_1x1][BUF_DPTH][BUF_DPTH],
				FIX_WT conv_weight_3x3_all[all_3x3][BUF_DPTH][3][3],
				FIX_WT bias_all[all_bias][BUF_DPTH],

				FIX_FM DDR_pool3_out_PL[l3_cd * BUF_DPTH][82][162],
				FIX_FM DDR_pool6_out_PL[l6_cd * BUF_DPTH][42][82],

				FIX_FM DDR_buf[36][BUF_DPTH][22][42],

				float predict_box[5]
)
{

#pragma HLS INTERFACE m_axi depth=3*162*322 	port=image_in_raw_pad			offset=slave	bundle=IMG
#pragma HLS INTERFACE m_axi depth=1181*16*16	port=conv_weight_1x1_all		offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=46*16*3*3		port=conv_weight_3x3_all		offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=123*16		port=bias_all					offset=slave	bundle=INPUT

#pragma HLS INTERFACE m_axi depth=48*82*162		port=DDR_pool3_out_PL			offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=96*42*82		port=DDR_pool6_out_PL			offset=slave	bundle=INPUT

#pragma HLS INTERFACE m_axi depth=36*16*22*42	port=DDR_buf					offset=slave	bundle=INPUT

#pragma HLS INTERFACE m_axi depth=5				port=predict_box				offset=slave	bundle=OUTPUT

#pragma HLS INTERFACE s_axilite register	port=return




#pragma HLS ALLOCATION instances=CONV_1x1			 		limit=1 function
#pragma HLS ALLOCATION instances=CONV_3x3_group     		limit=1 function
#pragma HLS ALLOCATION instances=max_pooling		    	limit=1 function
#pragma HLS ALLOCATION instances=load_image_chunk_norm		limit=1 function
#pragma HLS ALLOCATION instances=my_exp_fix					limit=1 function
#pragma HLS ALLOCATION instances=set_bias					limit=1 function
#pragma HLS ALLOCATION instances=Relu						limit=1 function
#pragma HLS ALLOCATION instances=load_weight_3D_from_axi	limit=1 function
#pragma HLS ALLOCATION instances=load_weight_2D_from_axi	limit=1 function


//#pragma HLS ARRAY_PARTITION variable=FM_bufs dim=1 block factor=16
#pragma HLS ARRAY_PARTITION variable=FM_bufs dim=2 block factor=16



	int weight_1x1_index = 0;
	int weight_3x3_index = 0;
	int bias_index = 0;


	int pre_weight_1x1_index = 0;
	int pre_weight_3x3_index = 0;
	int pre_bias_index = 0;

	/////////////////////////////// CONV_1 to POOL_3 ////////////////////////////

	load_weight_3D_from_axi(weight_buf_3x3_1, conv_weight_3x3_all[weight_3x3_index]);
	weight_3x3_index++;

	for(int row = 0; row < 8; row++) {
		for(int col = 0; col < 8; col++) {
//#pragma HLS unroll

			bias_index = 0;
			weight_1x1_index = 0;

			///// CONV_1 (3x3)  <---  IMG ch:0 col:{{_col}} row:{{_row}}
			load_image_chunk_norm(FM_bufs[0], image_in_raw_pad, col, row);
			load_bias_from_axi(bias_buf, bias_all[bias_index]);
			set_bias(FM_bufs[2], bias_buf);

			CONV_3x3_group(FM_bufs[0], FM_bufs[2], weight_buf_3x3_1);
			Relu(FM_bufs[2]);

			bias_index++;

#ifdef CSIM_DEBUG
fill_output(1, FM_bufs[2], 0, col, row);
#endif


#ifdef CSIM_DEBUG_FIX
fill_output_fix(1, FM_bufs[2], 0, col, row);
#endif



			for(int ch_conv2 = 0; ch_conv2 < l2_cd; ch_conv2++) {
//#pragma HLS unroll

				///// CONV_2 (1x1)  <---  CONV_1 ch:{{_ch_conv2}} col:{{_col}} row:{{_row}}
				load_weight_2D_from_axi(weight_buf_1x1_1, conv_weight_1x1_all[weight_1x1_index]);
				load_bias_from_axi(bias_buf, bias_all[bias_index]);
				set_bias(FM_bufs[14], bias_buf);

				CONV_1x1(FM_bufs[2], FM_bufs[14], weight_buf_1x1_1);
				Relu(FM_bufs[14]);

				///// POOL_3  <---  CONV_2 ch:{{_ch_conv2}} col:{{_col}} row:{{_row}}
				max_pooling(FM_bufs[14], FM_buf_pool);
				copy_to_DDR_pool3( DDR_pool3_out_PL, FM_buf_pool, ch_conv2, col, row);

				bias_index++;
				weight_1x1_index++;

#ifdef CSIM_DEBUG
	fill_output(2, FM_bufs[14], ch_conv2, col, row);
	fill_output_pool(3, FM_buf_pool, ch_conv2, col, row);
#endif

#ifdef CSIM_DEBUG_FIX
	fill_output_fix(2, FM_bufs[14], ch_conv2, col, row);
	fill_output_pool_fix(3, FM_buf_pool, ch_conv2, col, row);
#endif


			}
		}
	}


	pre_weight_1x1_index = weight_1x1_index;
	pre_weight_3x3_index = weight_3x3_index;
	pre_bias_index = bias_index;


	/////////////////////////////// CONV_4 to POOL_6  ////////////////////////////

	for(int row = 0; row < 4; row++) {
		for(int col = 0; col < 4; col++) {
//#pragma HLS unroll

			weight_1x1_index = pre_weight_1x1_index;
			weight_3x3_index = pre_weight_3x3_index;
			bias_index = pre_bias_index;

			for(int ch_conv4 = 0; ch_conv4 < l4_cd; ch_conv4++ ) {

				///// CONV_4  <---  POOL_3 ch:{{ch_conv4}} col:{{_col}} row:{{_row}}
				// load weight
				load_weight_3D_from_axi(weight_buf_3x3_1, conv_weight_3x3_all[weight_3x3_index]);
				// load bias
				load_bias_from_axi(bias_buf, bias_all[bias_index]);

				// load from DDR_pool_3_out_PL
				load_pool3_from_axi(FM_bufs[0], DDR_pool3_out_PL, ch_conv4, col, row);
				set_bias(FM_bufs[2 + ch_conv4], bias_buf);
				CONV_3x3_group(FM_bufs[0], FM_bufs[2 + ch_conv4], weight_buf_3x3_1);
				Relu(FM_bufs[2 + ch_conv4]);

				weight_3x3_index++;
				bias_index++;


#ifdef CSIM_DEBUG
fill_output(4, FM_bufs[2 + ch_conv4], ch_conv4, col, row);
#endif

#ifdef CSIM_DEBUG_FIX
fill_output_fix(4, FM_bufs[2 + ch_conv4], ch_conv4, col, row);
#endif

			}


			for(int ch_conv5 = 0; ch_conv5 < l5_cd; ch_conv5++) {
//#pragma HLS unroll

				load_bias_from_axi(bias_buf, bias_all[bias_index]);
				bias_index++;
				set_bias(FM_bufs[14], bias_buf);

				for(int ch_conv4 = 0; ch_conv4 < l4_cd; ch_conv4++ ) {
					load_weight_2D_from_axi(weight_buf_1x1_1, conv_weight_1x1_all[weight_1x1_index]);
					CONV_1x1(FM_bufs[2 + ch_conv4], FM_bufs[14], weight_buf_1x1_1);
					weight_1x1_index++;
				}

				Relu(FM_bufs[14]);

				///// POOL_6  <--- CONV_5 ch:{{_ch_conv5}} col:{{_col}} row:{{_row}}
				max_pooling(FM_bufs[14], FM_buf_pool);
				copy_to_DDR_pool6( DDR_pool6_out_PL, FM_buf_pool, ch_conv5, col, row);

#ifdef CSIM_DEBUG
	fill_output(5, FM_bufs[14], ch_conv5, col, row);
	fill_output_pool(6, FM_buf_pool, ch_conv5, col, row);
#endif

#ifdef CSIM_DEBUG_FIX
	fill_output_fix(5, FM_bufs[14], ch_conv5, col, row);
	fill_output_pool_fix(6, FM_buf_pool, ch_conv5, col, row);
#endif

			}
		}
	}

	pre_weight_1x1_index = weight_1x1_index;
	pre_weight_3x3_index = weight_3x3_index;
	pre_bias_index = bias_index;


	/////////////////////////////// CONV_7 to POOL_9  ////////////////////////////


	for(int col = 0; col < 2; col++) {
		for(int row = 0; row < 2; row++) {

			weight_1x1_index = pre_weight_1x1_index;
			weight_3x3_index = pre_weight_3x3_index;
			bias_index = pre_bias_index;


			for(int ch_conv7 = 0; ch_conv7 < l7_cd; ch_conv7++) {
//#pragma HLS unroll

				///// CONV_7  <--- POOL_6 ch:{{_ch_conv7}} col:{{_col}} row:{{_row}}
				load_weight_3D_from_axi(weight_buf_3x3_1, conv_weight_3x3_all[weight_3x3_index]);
				load_bias_from_axi(bias_buf, bias_all[bias_index]);
				load_pool6_from_axi(FM_bufs[0], DDR_pool6_out_PL, ch_conv7, col, row);

				set_bias(FM_bufs[2 + ch_conv7], bias_buf);
				CONV_3x3_group(FM_bufs[0], FM_bufs[2 + ch_conv7], weight_buf_3x3_1);
				Relu(FM_bufs[2 + ch_conv7]);
#ifdef CSIM_DEBUG
	fill_output(7, FM_bufs[2 + ch_conv7], ch_conv7, col, row);
#endif

#ifdef CSIM_DEBUG_FIX
	fill_output_fix(7, FM_bufs[2 + ch_conv7], ch_conv7, col, row);
#endif

				weight_3x3_index++;
				bias_index++;
			}

			for(int ch_conv8 = 0; ch_conv8 < l8_cd; ch_conv8++ ) {
//#pragma HLS unroll

				///// CONV_8  <--- CONV_7 ch:{{_ch_conv8}} col:{{_col}} row:{{_row}}
				load_bias_from_axi(bias_buf, bias_all[bias_index]);
				bias_index++;
				set_bias(FM_bufs[14], bias_buf);

				for(int ch_conv7 = 0; ch_conv7 < l7_cd; ch_conv7++ ) {
					load_weight_2D_from_axi(weight_buf_1x1_1, conv_weight_1x1_all[weight_1x1_index]);
					CONV_1x1(FM_bufs[2 + ch_conv7], FM_bufs[14], weight_buf_1x1_1);
					weight_1x1_index++;
				}

				Relu(FM_bufs[14]);

				///// POOL_9  <--- CONV_8 ch:{{_ch_conv8}} col:{{_col}} row:{{_row}}
				max_pooling(FM_bufs[14], FM_buf_pool);
				copy_to_DDR_pool9( DDR_buf[ch_conv8], FM_buf_pool, col, row);

#ifdef CSIM_DEBUG
	fill_output(8, FM_bufs[14], ch_conv8, col, row);
	fill_output_pool(9, FM_buf_pool, ch_conv8, col, row);
#endif

#ifdef CSIM_DEBUG_FIX
	fill_output_fix(8, FM_bufs[14], ch_conv8, col, row);
	fill_output_pool_fix(9, FM_buf_pool, ch_conv8, col, row);
#endif


			}
		}
	}

	pre_weight_1x1_index = weight_1x1_index;
	pre_weight_3x3_index = weight_3x3_index;
	pre_bias_index = bias_index;


	/////////////////////////////// CONV_10 to CONV_11  //////////////////////////

	for(int col = 0; col < 1; col++) {
		for(int row = 0; row < 1; row++) {

			weight_1x1_index = pre_weight_1x1_index;
			weight_3x3_index = pre_weight_3x3_index;
			bias_index = pre_bias_index;


			for(int ch_conv10 = 0; ch_conv10 < l10_cd; ch_conv10++ ) {
//#pragma HLS unroll


				///// CONV_10  <--- POOL_9 ch:{{_ch_conv10}} col:{{_col}} row:{{_row}}
				load_weight_3D_from_axi(weight_buf_3x3_1, conv_weight_3x3_all[weight_3x3_index]);
				load_bias_from_axi(bias_buf, bias_all[bias_index]);
				buffer_copy_from_axi(FM_bufs[0], DDR_buf[ch_conv10]);

				set_bias(FM_bufs[2 + ch_conv10], bias_buf);
				CONV_3x3_group(FM_bufs[0], FM_bufs[2 + ch_conv10], weight_buf_3x3_1);
				Relu(FM_bufs[2 + ch_conv10]);

				weight_3x3_index++;
				bias_index++;

#ifdef CSIM_DEBUG
	fill_output(10, FM_bufs[2 + ch_conv10], ch_conv10, col, row);
#endif

#ifdef CSIM_DEBUG_FIX
	fill_output_fix(10, FM_bufs[2 + ch_conv10], ch_conv10, col, row);
#endif
			}

			for(int ch_conv11 = 0; ch_conv11 < l11_cd; ch_conv11++ ) {
//#pragma HLS unroll

				///// CONV_11  <--- CONV_10 ch:{{_ch_conv11}} col:{{_col}} row:{{_row}}
				load_bias_from_axi(bias_buf, bias_all[bias_index]);
				bias_index++;
				set_bias(FM_bufs[14], bias_buf);

				for(int ch_conv10 = 0; ch_conv10 < l10_cd; ch_conv10++) {
//#pragma HLS unroll
					load_weight_2D_from_axi(weight_buf_1x1_1, conv_weight_1x1_all[weight_1x1_index]);
					CONV_1x1(FM_bufs[2 + ch_conv10], FM_bufs[14], weight_buf_1x1_1);
					weight_1x1_index++;
				}

				Relu(FM_bufs[14]);
				buffer_copy_to_axi(DDR_buf[ch_conv11], FM_bufs[14]);

#ifdef CSIM_DEBUG
	fill_output(11, FM_bufs[14], ch_conv11, col, row);
#endif

#ifdef CSIM_DEBUG_FIX
	fill_output_fix(11, FM_bufs[14], ch_conv11, col, row);
#endif

			}
		}
	}


	/////////////////////////////// CONV_12 //////////////////////////
	for(int ch_conv12 = 0; ch_conv12 < l12_cd; ch_conv12++) {
//#pragma HLS unroll

		buffer_copy_from_axi(FM_bufs[0], DDR_buf[ch_conv12]);
		load_weight_3D_from_axi(weight_buf_3x3_1, conv_weight_3x3_all[weight_3x3_index]);
		load_bias_from_axi(bias_buf, bias_all[bias_index]);

		set_bias(FM_bufs[2], bias_buf);
		CONV_3x3_group(FM_bufs[0], FM_bufs[2], weight_buf_3x3_1);
		Relu(FM_bufs[2]);
		buffer_copy_to_axi(DDR_buf[ch_conv12], FM_bufs[2]);


		weight_3x3_index++;
		bias_index++;

#ifdef CSIM_DEBUG
	fill_output(12, FM_bufs[2], ch_conv12, 0, 0);
#endif

#ifdef CSIM_DEBUG_FIX
	fill_output_fix(12, FM_bufs[2], ch_conv12, 0, 0);
#endif

	}


	/////////////////////////////// CONV_13 to CONV_14  //////////////////////////

	clear_buf(FM_bufs[15]);
	for(int ch_conv13 = 0; ch_conv13 < l13_cd; ch_conv13++) {
//#pragma HLS unroll


		load_bias_from_axi(bias_buf, bias_all[bias_index]);
		bias_index++;
		set_bias(FM_bufs[14], bias_buf);


		for(int ch_conv12 = 0; ch_conv12 < l12_cd; ch_conv12++) {
//#pragma HLS unroll
			load_weight_2D_from_axi(weight_buf_1x1_1, conv_weight_1x1_all[weight_1x1_index]);
			buffer_copy_from_axi(FM_bufs[2], DDR_buf[ch_conv12]);
			CONV_1x1(FM_bufs[2], FM_bufs[14], weight_buf_1x1_1);
			weight_1x1_index++;
		}


		Relu(FM_bufs[14]);


#ifdef CSIM_DEBUG
	fill_output(13, FM_bufs[14], ch_conv13, 0, 0);
#endif


#ifdef CSIM_DEBUG_FIX
	fill_output_fix(13, FM_bufs[14], ch_conv13, 0, 0);
#endif


		load_weight_2D_from_axi(weight_buf_1x1_3, conv_weight_1x1_all[all_1x1 - l13_cd + ch_conv13]);
		CONV_1x1(FM_bufs[14], FM_bufs[15], weight_buf_1x1_3);

	}


#ifdef CSIM_DEBUG
	fill_output(14, FM_bufs[15], 0, 0, 0);
	PL_golden_compare_layer_1();
	PL_golden_compare_layer_2();
	PL_golden_compare_layer_3();
	PL_golden_compare_layer_4();
	PL_golden_compare_layer_5();
	PL_golden_compare_layer_6();
	PL_golden_compare_layer_7();
	PL_golden_compare_layer_8();
	PL_golden_compare_layer_9();
	PL_golden_compare_layer_10();
	PL_golden_compare_layer_11();
	PL_golden_compare_layer_12();
	PL_golden_compare_layer_13();
	PL_golden_compare_layer_14();
#endif


#ifdef CSIM_DEBUG_FIX
	fill_output_last_fix(14, FM_bufs[15], 0, 0, 0);
	output_PL_layers();
#endif


	compute_bounding_box(predict_box);

	return;



}
