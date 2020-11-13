#include "conv.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(){
	static axis_t map_in[128*34*34];
	static axis_t bias[128];
	static axis_t weight_kernel[128*128*3*3];
	static axis_t map_out[128*32*32];
	static feature_map map_in_buffer[128*34*34];
	static feature_map bias_buffer[128];
	static feature_map weight_kernel_buffer[128*128*3*3];

	ifstream in("C:/Users/10720/Desktop/FPGA/HLS/conv_add/map_in.data", ios::in | ios::binary);
	in.read((char *) &map_in_buffer, sizeof map_in_buffer);
	// see how many bytes have been readz
	cout << in.gcount() << " bytes read with map_in\n";
	in.close();

	ifstream w_kernel("C:/Users/10720/Desktop/FPGA/HLS/conv_add/weight.data", ios::in | ios::binary);
	w_kernel.read((char *) &weight_kernel_buffer, sizeof weight_kernel_buffer);
	// see how many bytes have been readz
	cout << "\n" << w_kernel.gcount() << " bytes read with weight_kernel\n";
	w_kernel.close();

	ifstream b("C:/Users/10720/Desktop/FPGA/HLS/conv_add/bias.data", ios::in | ios::binary);
	b.read((char *) &bias_buffer, sizeof bias_buffer);
	// see how many bytes have been readz
	cout << "\n" << b.gcount() << " bytes read with bias\n";
	b.close();

	int C = 34;
	int R = 34;
	int CHI = 128;
	int CHO = 32;

	for(int i=0; i<128*34*34; i++){
		map_in[i].data = map_in_buffer[i];
	}

	for(int i=0; i<128*128*3*3; i++){
		weight_kernel[i].data = weight_kernel_buffer[i];
	}

	for(int i=0; i<128; i++){
		bias[i].data = bias_buffer[i];
	}

	cout << '\n' << "map_in: \n";
	for(int i=C-10; i<C; i++){
		cout << map_in[(CHI-1)*34*34+(R-2)*34+i].data << " ";
	}
	cout << "\n";

	cout << '\n' << "bias: \n";
	for(int i=CHO-10; i<CHO; i++){
		cout << bias[i].data << " ";
	}
	cout << "\n";

	cout << "weight_kernel: \n";
	for(int i=0; i<3; i++){
		cout << '\n';
		for(int j=0; j<3; j++){
			cout << weight_kernel[(CHI-1)*128*3*3+(CHO-1)*3*3+i*3+j].data << " ";
		}
	}
	cout << "\n";


	int ctr = 1;

	top(map_in,weight_kernel,bias,map_out,ctr);

	cout << "map_out: \n";
	for(int i=22; i<32; i++){
		cout << map_out[i*32*32+31*32+31].data << " ";
	}

	cout << "\n";

	return 0;
}
