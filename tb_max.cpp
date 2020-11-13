#include "conv.h"
#include <iostream>
#include <math.h>
using namespace std;

int main(){
	static axis_t map_in[64*34*34];
	static axis_t bias[32];
	static axis_t weight[64*32*3*3];
	static axis_t map_out[64*32*32];
	static half out[64][32][32];

	for (int chi=0; chi < 64; chi++){
			for(int r=0; r<34; r++){
				for(int c=0; c<34; c++){
					map_in[chi*34*34+r*34+c].data = r*34 + c;
				}
			}
		}

	int ctr=4;
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
	for(int r=0; r<32; r++){
		cout << '\n';
		for(int c=0; c<32; c++){
			cout << map_in[44*34*34+r*34+c].data << " ";
		}
	}

	cout << "\n================================== after =================================\n";
	for(int r=0; r<16; r++){
		cout << '\n';
		for(int c=0; c<16; c++){
			cout << out[44][r][c] << " ";
		}
	}

	return 0;
}
