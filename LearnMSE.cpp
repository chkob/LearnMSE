// LearnMSE.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NeuralNet.h"

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	NeuralNet net;
 
	float x[1000], y[1000];
		int sp[1000];
	for(int i = 0; i < 1000; i++) {
		x[i] = i+1;
		y[i] = log(x[i]) + pow(x[i], 3) - 123;
		
		//x[i] /= 1000;
		//y[i] /= 1e8;
		
		sp[i] = i;
	}

	int layerNeuronsNum[3] = { 1, 20, 1 };
	net.CreateNetwork(3, layerNeuronsNum);
	net.SetLearningRate(0.01f);
	net.SetMomentum(0.05f);

	//for (int it = 0; it < 200; it ++) {
	//	std::random_shuffle(sp, sp+1000);
	//	for (int i = 0; i < 1000; i++) {
	//		vector<float> in, out;
	//		in.push_back(x[sp[i]]);	out.push_back(y[sp[i]]);
	//		net.OnlineTrain(in, out, false);
	//	}
	//}

	//for (int i = 0; i < 1000; i++) {
	//		vector<float> in, out;
	//		in.push_back(x[i]);	
	//		//..net.OnlineTrain(in, out, false);
	//		net.Run(in, out);
	//		std::cout << out[0] << " " << y[i] << std::endl;
	//	}
	//net.SaveNetwork("net");

	std::random_shuffle(sp, sp+1000);

	vector<vector<float> > inList, outList;
	for (int i = 0; i < 1000; i++) {
		vector<float> in, out;
		in.push_back(x[sp[i]]);	out.push_back(y[sp[i]]);
		inList.push_back(in);	outList.push_back(out);
	}

	net.BatchTrain(inList, outList, 100000);

	for (int i = 0; i < 1000; i++) {
			vector<float> in, out;
			in.push_back(x[i]);	
			//..net.OnlineTrain(in, out, false);
			net.Run(in, out);
			std::cout << out[0] << " " << y[i] << std::endl;
		}


	return 0;
}

