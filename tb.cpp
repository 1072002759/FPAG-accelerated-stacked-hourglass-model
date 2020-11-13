#include "conv.h"
#include <iostream>
#include <math.h>
using namespace std;

int main(){
	static axis_t map_in[128*34*34];
	static axis_t bias[64];
	static axis_t weight[128*64*3*3];
	static axis_t map_out[64*32*32];
	static half out[64][32][32];

	for(int i=0; i<128*34*34; i++){
		map_in[i].data = (half)((rand()%10)*0.001);
	}

	for(int i=0; i<128*64*3*3; i++){
		weight[i].data = (half)((rand()%10)*0.1*pow(-1,(rand()%10)));
	}

	for(int i; i<64; i++){
		bias[i].data = (half)0.01;
	}


	int ctr=1;
	top(map_in,weight,bias,map_out,map_in,weight,bias,map_out,ctr);

	int p = 0;
	for (int cho=0; cho < 64; cho++){
		for(int r=0; r<32; r++){
			for(int c=0; c<32; c++){
				out[cho][r][c] = map_out[p++].data;
			}
		}
	}

	cout << "\n================================== 8 =================================\n";
	for(int i=0; i<8; i++){
		cout << out[63][7][i] << "\n";
	}

	cout << "\n================================== 8_cho =================================\n";
	for(int i=53; i<64; i++){
		cout << out[i][7][7] << "\n";
	}

	cout << "\n================================== 32 =================================\n";
	for(int i=7; i<18; i++){
		cout << out[31][8][i] << "\n";
	}

	cout << "\n================================== 32_cho =================================\n";
	for(int i=30; i<33; i++){
		cout << out[i][8][8] << "\n";
	}

	cout << "\n================================== 68 =================================\n";
	for(int i=53; i<64; i++){
		cout << out[i][32][32] << "\n";
	}

	return 0;
}
