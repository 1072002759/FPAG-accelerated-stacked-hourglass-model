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

	for(int i=0; i<64*34*34; i++){
		map_in[i].data = i;
	}

	int ctr=5;
	top(map_in,weight,bias,map_out,map_in,weight,bias,map_out,ctr);

	int p = 0;
	for (int cho=0; cho < 64; cho++){
		for(int r=0; r<32; r++){
			for(int c=0; c<32; c++){
				out[cho][r][c] = map_out[p++].data;
			}
		}
	}

	cout << "\n================================== before =================================\n";
	for(int r=0; r<16; r++){
		cout << '\n';
		for(int c=0; c<16; c++){
			cout << map_in[55*34*34+r*34+c].data << " ";
		}
	}

	cout << "\n================================== after =================================\n";
	for(int r=0; r<32; r++){
		cout << '\n';
		for(int c=0; c<32; c++){
			cout << out[55][r][c] << " ";
		}
	}

	return 0;
}
