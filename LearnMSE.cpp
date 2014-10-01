// LearnMSE.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NeuralNet.h"

std::string hidden_choice;

float c;

int TEST_N;

int _tmain(int argc, _TCHAR* argv[])
{

	//////* Setting the GPU device*////
	int devID = 0;
	cudaError_t error;
	cudaDeviceProp deviceProp;
	GpuErrorCheck(cudaGetDevice(&devID));
	cudaFree(0); // context establishment happens here. This initialize GPU.

	TEST_N = 50;

	NeuralNet net;
	 
	int layerNeuronsNum[4] = { 41, 30, 20, 1 };

	hidden_choice = "_"+to_string(layerNeuronsNum[1])+"_"+to_string(layerNeuronsNum[2])+"_" + "RelMSE_TansigRect";

	net.CreateNetwork(4, layerNeuronsNum);
	
	std::vector<std::pair<std::pair<int, int>, std::string> > fileLists, testFileLists;


	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(750-10,1000-10), "chess"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(512-10,512-10), "cornellSphere"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1000-10,500-10), "dof-dragons"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "dragonfog"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "plants-dusk"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1000-10,1000-10), "sponza-fog"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "yeahright_nn"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(512-10,512-10), "toasters"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(800-10,1200-10), "sanmiguel"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(800-10,1200-10), "sanmiguel_cam4"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(600-10,600-10), "GlossyBalls"));
	fileLists.push_back(std::pair<std::pair<int, int>, std::string>(std::pair<int,int>(1000-10,500-10), "anim-bluespheres"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(800-10,800-10), "avatar"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(900-10,1280-10), "BuddhaMB"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(600-10,600-10), "DOF_Quads_Front"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(600-10,600-10), "DOF_Quads_Middle"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1000-10,500-10), "Liftoff_MB_DOF_nn"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(512-10,512-10), "MB_MovingSphere"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(512-10,512-10), "MB_MovingSphereStriped"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1600-10,1200-10), "sponza_floor"));
	fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1600-10,1200-10), "sponza_floor_MB"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1200-10,800-10), "room-path"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "conference"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(800-10,1200-10), "sanmiguel_cam1"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(800-10,1200-10), "sanmiguel_cam5"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(800-10,800-10), "teapot-metal"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "poolball"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "killeroos_nn"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "sibenik"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1024-10,1024-10), "sanmiguel20"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1600-10,1200-10), "sanmiguel_cam25"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1368-10,1026-10), "killeroo-gold"));
	//fileLists.push_back(std::pair<std::pair<int,int>, std::string>(std::pair<int,int>(1600-10,1200-10), "crytek_sponza"));

	
	/*net.GPU_CopyWeights();
	net.TestNN(fileLists);*/

	const int sampelsNum = 5, max_epochs = 100000;
	const unsigned long long wholeDataSize = 1000 * 1000 * 24/** 11*/;
	net.BatchTrain(fileLists, testFileLists, sampelsNum, max_epochs, wholeDataSize);
	net.SaveNetwork("Learn.net");

	cudaDeviceReset();
	// Dump memory leaks
	_CrtDumpMemoryLeaks();

	return 0;
}