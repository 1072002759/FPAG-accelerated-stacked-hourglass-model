#include "conv.h"
#include <iostream>
#include <math.h>
using namespace std;

int main(){
	static axis_t map_in[64*34*34];
	static axis_t bias[64];
	static axis_t weight[64*64*3*3];
	static axis_t map_out[64*32*32];
	static half out[64][32][32];

	for(int i=0; i<9; i++){
		map_in[i].data = i+i*0.1;
	}

	int ctr=7;
	top(map_in,weight,bias,map_out,map_in,weight,bias,map_out,ctr);

	int p = 0;
	for (int cho=0; cho < 64; cho++){
		for(int r=0; r<32; r++){
			for(int c=0; c<32; c++){
				out[cho][r][c] = map_out[p++].data;
			}
		}
	}

	for(int i=0; i<9; i++){
		cout << out[0][0][i] << '\n';
	}

	return 0;
}
