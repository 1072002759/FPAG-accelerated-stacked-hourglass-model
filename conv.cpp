#include "conv.h"
using namespace std;

void top(axis_t* In_ddr, axis_t* W_ddr, axis_t* B_ddr, axis_t* Out_m_ddr,axis_t* In_ddr_2, axis_t* W_ddr_2, axis_t* B_ddr_2, axis_t* Out_m_ddr_2, int ctr){
#pragma HLS INTERFACE s_axilite port=ctr
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE axis register both depth=64 port=B_ddr_2
#pragma HLS INTERFACE axis register both depth=13824 port=W_ddr_2
#pragma HLS INTERFACE axis register both depth=27744 port=In_ddr_2
#pragma HLS INTERFACE axis register both depth=65536 port=Out_m_ddr_2

#pragma HLS INTERFACE axis register both depth=64 port=B_ddr
#pragma HLS INTERFACE axis register both depth=65536 port=Out_m_ddr
#pragma HLS INTERFACE axis register both depth=36864 port=W_ddr
#pragma HLS INTERFACE axis register both depth=73984 port=In_ddr

	static feature_map In[64][34][34];
#pragma HLS ARRAY_PARTITION variable=In cyclic factor=32 dim=1
	static weight_type W[64][64][3][3];
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=32 dim=1
	static feature_map Out_m[64][32][32];
	static feature_map B[64];

	static feature_map In_2[3][68][68];
	static weight_type W_2[3][64][6][6];
#pragma HLS ARRAY_PARTITION variable=W_2 cyclic factor=32 dim=2
	static feature_map Out_m_2[64][32][32];
#pragma HLS ARRAY_PARTITION variable=Out_m_2 cyclic factor=32 dim=1
	static feature_map B_2[64];

	int p=0;
	int temp=0;

	switch(ctr){

//-----------------------------------------------------------------------------------------------------------
//################################################ conv_8_8 #################################################
//-----------------------------------------------------------------------------------------------------------

	case 1:
		//============================================= Load_B =================================================

			p=0;
			for(int cho=0; cho<64; cho++){
#pragma HLS PIPELINE II=1
				B[cho] = B_ddr[p++].data;
			}

			p=0;
			for(int cho=0; cho<64; cho++){
				for(int r=0; r<32; r++){
					for(int c=0; c<32; c++){
		#pragma HLS PIPELINE II=1
						Out_m[cho][r][c] = B[cho];
					}
				}
			}


			for (int i=0; i<128; i+=64){

			//============================================= Load_W =================================================

				p = i*64*3*3;
				for (int chi=0; chi < 64; chi++){
					for(int cho=0; cho<64; cho++){
						for(int kr=0; kr<3; kr++){
							for(int kc=0; kc<3; kc++){

			#pragma HLS PIPELINE II=1
								W[chi+i][cho][kr][kc] = W_ddr[p++].data;
							}
						}
					}
				}
			//============================================= Load_In =================================================

				p=i*34*34;
				for (int chi=0; chi < 64; chi++){
					for(int r=0; r<34; r++){
						for(int c=0; c<34; c++){
			#pragma HLS PIPELINE II=1
							In[chi+i][r][c] = In_ddr[p++].data;
						}
					}
				}

			//================================================= CONV 8x8 ===========================================================

				for(int kr=0; kr < 3; kr++){
					for(int kc=0; kc < 3; kc++){
						for(int r=0; r < 8; r++){
							for(int c=0; c < 8; c++){
								for(int cho=0; cho < 64; cho++){
		#pragma HLS PIPELINE II=1
									for(int chi=0; chi < 64; chi++){
												Out_m[cho][r][c] += In[chi+i][r+kr][c+kc] * W[chi+i][cho][kr][kc];
									}
								}
							}
						}
					}
				}
			}

		//========================================= OFF_LOAD OUT ====================================

		temp=0;
		p=0;
		for (int cho=0; cho<64; cho++){
			for(int r=0; r<32; r++){
				for(int c=0; c<32; c++){
	#pragma HLS PIPELINE II=1
					if(cho==63 && r==31 && c==31){
						temp = 1;
					}
					else{
						temp = 0;
					}
					Out_m_ddr[p].data = Out_m[cho][r][c]>(half)0 ? Out_m[cho][r][c] : (half)0;
					Out_m_ddr[p++].last = temp;
				}
			}
		}


		break;

//-----------------------------------------------------------------------------------------------------------
//############################################### conv_32_32 ################################################
//-----------------------------------------------------------------------------------------------------------

	case 2:
		//============================================= Load_B =================================================

			p=0;
			for(int cho=0; cho<64; cho++){
#pragma HLS PIPELINE II=1
				B[cho] = B_ddr[p++].data;
			}

			p=0;
			for(int cho=0; cho<64; cho++){
				for(int r=0; r<32; r++){
					for(int c=0; c<32; c++){
		#pragma HLS PIPELINE II=1
						Out_m[cho][r][c] = B[cho];
					}
				}
			}


		for(int i=0; i<128; i+=64){
			//============================================= Load_W =================================================

			p = i*64*3*3;
			for (int chi=0; chi < 64; chi++){
				for(int cho=0; cho<64; cho++){
					for(int kr=0; kr<3; kr++){
						for(int kc=0; kc<3; kc++){

		#pragma HLS PIPELINE II=1
							W[chi+i][cho][kr][kc] = W_ddr[p++].data;
						}
					}
				}
			}

			//============================================= Load_In =================================================

			p=i*34*34;
			for (int chi=0; chi < 64; chi++){
				for(int r=0; r<34; r++){
					for(int c=0; c<34; c++){
		#pragma HLS PIPELINE II=1
						In[chi+i][r][c] = In_ddr[p++].data;
					}
				}
			}
			//================================================= CONV 32x32 ===========================================================

				kernel_r_1:
				for(int kr=0; kr < 3; kr++){
					kernel_c_1:
					for(int kc=0; kc < 3; kc++){
						Row_1:
						for(int r=0; r < 32; r++){
							Col_1:
							for(int c=0; c < 32; c++){
								Output_Channel_1:
								for(int cho=0; cho < 64; cho++){
		#pragma HLS PIPELINE II=1
									Input_Channel_1:
									for(int chi=0; chi < 64; chi++){
												Out_m[cho][r][c] += In[chi+i][r+kr][c+kc] * W[chi+i][cho][kr][kc];
									}
								}
							}
						}
					}
				}
			}

		//========================================= OFF_LOAD OUT ====================================

		temp=0;
		p=0;
		for (int cho=0; cho<64; cho++){
			for(int r=0; r<32; r++){
				Offload_Out_m_label3:
				for(int c=0; c<32; c++){
	#pragma HLS PIPELINE II=1
					if(cho==63 && r==31 && c==31){
						temp = 1;
					}
					else{
						temp = 0;
					}
					Out_m_ddr[p].data = Out_m[cho][r][c]>(half)0 ? Out_m[cho][r][c] : (half)0;
					Out_m_ddr[p++].last = temp;
				}
			}
		}


		break;

//-----------------------------------------------------------------------------------------------------------
//############################################## conv_68_68 ###############################################
//-----------------------------------------------------------------------------------------------------------

		case 3:
	//============================================= Load_B =================================================

			p=0;
			for(int cho=0; cho<64; cho++){
#pragma HLS PIPELINE II=1
				B_2[cho] = B_ddr_2[p++].data;
			}

			p=0;
			for(int cho=0; cho<64; cho++){
				for(int r=0; r<32; r++){
					for(int c=0; c<32; c++){
		#pragma HLS PIPELINE II=1
						Out_m_2[cho][r][c] = B_2[cho];
					}
				}
			}


	//============================================= Load_W =================================================

			p = 0;
			for (int chi=0; chi < 3; chi++){
				for(int cho=0; cho<64; cho++){
					for(int kr=0; kr<6; kr++){
						for(int kc=0; kc<6; kc++){
		#pragma HLS PIPELINE II=1
							W_2[chi][cho][kr][kc] = W_ddr_2[p++].data;
						}
					}
				}
			}

	//============================================= Load_In =================================================

			p=0;
			for (int chi=0; chi < 3; chi++){
				for(int r=0; r<68; r++){
					for(int c=0; c<68; c++){
		#pragma HLS PIPELINE II=1
						In_2[chi][r][c] = In_ddr_2[p++].data;
					}
				}
			}
	//================================================= CONV 132x132 ===========================================================

		for(int kr=0; kr < 6; kr++){
			for(int kc=0; kc < 6; kc++){
				for(int r=0; r < 32; r++){
					for(int c=0; c < 32; c++){
						for(int chi=0; chi < 3; chi++){
	#pragma HLS PIPELINE II=6
							for(int cho=0; cho < 64; cho++){
										Out_m_2[cho][r][c] += In_2[chi][2*r+kr][2*c+kc] * W_2[chi][cho][kr][kc];
								}
							}
						}
					}
				}
			}


	//========================================= OFF_LOAD OUT ====================================

		temp=0;
		p=0;
		for (int cho=0; cho<64; cho++){
			for(int r=0; r<32; r++){
				for(int c=0; c<32; c++){
	#pragma HLS PIPELINE II=1
					if(cho==63 && r==31 && c==31){
						temp = 1;
					}
					else{
						temp = 0;
					}
					Out_m_ddr_2[p].data = Out_m_2[cho][r][c]>(half)0 ? Out_m_2[cho][r][c] : (half)0;
					Out_m_ddr_2[p++].last = temp;
					}
				}
			}


		break;

//-----------------------------------------------------------------------------------------------------------
//############################################### MAX POOL ################################################
//-----------------------------------------------------------------------------------------------------------

		case 4:

		//============================================= Load_In =================================================

			p=0;
			for (int chi=0; chi < 64; chi++){
				for(int r=0; r<34; r++){
					for(int c=0; c<34; c++){
		#pragma HLS PIPELINE II=1
						In[chi][r][c] = In_ddr[p++].data;
					}
				}
			}

		//================================================ SET SMALL =================================================

			for(int cho=0; cho<64; cho++){
				for(int r=0; r<16; r++){
					for(int c=0; c<16; c++){
#pragma HLS PIPELINE II=1
						Out_m[cho][r][c] = (half)-2000.1;
					}
				}
			}

		//================================================= CONV 32x32 ===========================================================

			for(int ch=0; ch < 64; ch++){
				for(int r=0; r < 16; r++){
					top_label0:
					for(int c=0; c < 16; c++){
						for(int kr=0; kr < 2; kr++){
							for(int kc=0; kc < 2; kc++){
#pragma HLS PIPELINE II=3
								Out_m[ch][r][c] = In[ch][(2*r)+kr][(2*c)+kc]>Out_m[ch][r][c] ? In[ch][(2*r)+kr][(2*c)+kc] : Out_m[ch][r][c];
							}
						}
					}
				}
			}

			cout << "\n test \n" << Out_m[0][0][0];
		//========================================= OFF_LOAD OUT ====================================

			temp=0;
			p=0;
			for (int cho=0; cho<64; cho++){
				for(int r=0; r<32; r++){
					for(int c=0; c<32; c++){
		#pragma HLS PIPELINE II=1
						if(cho==63 && r==31 && c==31){
							temp = 1;
						}
						else{
							temp = 0;
						}
						Out_m_ddr[p].data = Out_m[cho][r][c];
						Out_m_ddr[p++].last = temp;
					}
				}
			}


			break;

//-----------------------------------------------------------------------------------------------------------
//################################################# UP UP ###################################################
//-----------------------------------------------------------------------------------------------------------

		case 5:

		//============================================= Load_In =================================================

			p=0;
			for (int chi=0; chi < 64; chi++){
				for(int r=0; r<34; r++){
					for(int c=0; c<34; c++){
		#pragma HLS PIPELINE II=1
						In[chi][r][c] = In_ddr[p++].data;
					}
				}
			}

		//================================================= CONV 32x32 ===========================================================

			for(int r=0; r < 16; r++){
				for(int c=0; c < 16; c++){
					for(int ch=0; ch < 64; ch++){
#pragma HLS PIPELINE II=2
						Out_m[ch][2*r][2*c] = In[ch][r][c];
						Out_m[ch][2*r+1][2*c] = In[ch][r][c];
						Out_m[ch][2*r][2*c+1] = In[ch][r][c];
						Out_m[ch][2*r+1][2*c+1] = In[ch][r][c];
					}
				}
			}

		//========================================= OFF_LOAD OUT ====================================

			temp=0;
			p=0;
			for (int cho=0; cho<64; cho++){
				for(int r=0; r<32; r++){
					for(int c=0; c<32; c++){
		#pragma HLS PIPELINE II=1
						if(cho==63 && r==31 && c==31){
							temp = 1;
						}
						else{
							temp = 0;
						}
						Out_m_ddr[p].data = Out_m[cho][r][c];
						Out_m_ddr[p++].last = temp;
					}
				}
			}


			break;

//-----------------------------------------------------------------------------------------------------------
//######################################### conv_32_32_without_rulu ##########################################
//-----------------------------------------------------------------------------------------------------------

	case 6:
		//============================================= Load_B =================================================

			p=0;
			for(int cho=0; cho<64; cho++){
#pragma HLS PIPELINE II=1
				B[cho] = B_ddr[p++].data;
			}

			p=0;
			for(int cho=0; cho<64; cho++){
				for(int r=0; r<32; r++){
					for(int c=0; c<32; c++){
		#pragma HLS PIPELINE II=1
						Out_m[cho][r][c] = B[cho];
					}
				}
			}



		for(int i=0; i<128; i+=64){
			//============================================= Load_W =================================================

			p = i*64*3*3;
			for (int chi=0; chi < 64; chi++){
				for(int cho=0; cho<64; cho++){
					for(int kr=0; kr<3; kr++){
						for(int kc=0; kc<3; kc++){

		#pragma HLS PIPELINE II=1
							W[chi+i][cho][kr][kc] = W_ddr[p++].data;
						}
					}
				}
			}

			//============================================= Load_In =================================================

			p=i*34*34;
			for (int chi=0; chi < 64; chi++){
				for(int r=0; r<34; r++){
					for(int c=0; c<34; c++){
		#pragma HLS PIPELINE II=1
						In[chi+i][r][c] = In_ddr[p++].data;
					}
				}
			}
			//================================================= CONV 32x32 ===========================================================

				for(int kr=0; kr < 3; kr++){
					for(int kc=0; kc < 3; kc++){
						for(int r=0; r < 32; r++){
							for(int c=0; c < 32; c++){
								for(int cho=0; cho < 64; cho++){
		#pragma HLS PIPELINE II=1
									for(int chi=0; chi < 64; chi++){
												Out_m[cho][r][c] += In[chi+i][r+kr][c+kc] * W[chi+i][cho][kr][kc];
									}
								}
							}
						}
					}
				}
			}

		//========================================= OFF_LOAD OUT ====================================

		temp=0;
		p=0;
		for (int cho=0; cho<64; cho++){
			for(int r=0; r<32; r++){
				for(int c=0; c<32; c++){
	#pragma HLS PIPELINE II=1
					if(cho==63 && r==31 && c==31){
						temp = 1;
					}
					else{
						temp = 0;
					}
					Out_m_ddr[p].data = Out_m[cho][r][c];
					Out_m_ddr[p++].last = temp;
				}
			}
		}


		break;

//-----------------------------------------------------------------------------------------------------------
//################################################### sigmoid ###############################################
//-----------------------------------------------------------------------------------------------------------
	case 7:
//============================================= Load_In =================================================

		p=0;
		for (int chi=0; chi < 64; chi++){
			for(int r=0; r<34; r++){
				for(int c=0; c<34; c++){
	#pragma HLS PIPELINE II=1
					In[chi][r][c] = In_ddr[p++].data;
				}
			}
		}

//================================================= Sigmoid ===================================================

		for(int ch=0; ch<16; ch++){
			for(int r=0; r<32; r++){
				for(int c=0; c<32; c++){
#pragma HLS PIPELINE II=1
					Out_m[ch][r][c] = 1/(1+exp(-In[ch][r][c]));
				}
			}
		}

//========================================= OFF_LOAD OUT ====================================

		temp=0;
		p=0;
		for (int cho=0; cho<64; cho++){
			for(int r=0; r<32; r++){
				for(int c=0; c<32; c++){
	#pragma HLS PIPELINE II=1
					if(cho==63 && r==31 && c==31){
						temp = 1;
					}
					else{
						temp = 0;
					}
					Out_m_ddr[p].data = Out_m[cho][r][c];
					Out_m_ddr[p++].last = temp;
				}
			}
		}
		break;

	}


}

