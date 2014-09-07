// LearnMSE.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NeuralNet.h"

std::string hidden_choice;

int _tmain(int argc, _TCHAR* argv[])
{

	//* Setting the GPU device*////
	int devID = 0;
	cudaError_t error;
	cudaDeviceProp deviceProp;
	GpuErrorCheck(cudaGetDevice(&devID));
	cudaFree(0); // context establishment happens here. This initialize GPU.



	NeuralNet net;
	
	int layerNeuronsNum[4] = { 37, 45, 35, 1 };

	hidden_choice = "_45_35_lr0.001_";

	net.CreateNetwork(4, layerNeuronsNum);

	//net.LoadNetwork("final_all_40_30_lr0.001_.net");

	std::vector<std::pair<std::string, std::string> > fileLists;

	fileLists.push_back(std::pair<std::string, std::string>("chess", "chess"));
	fileLists.push_back(std::pair<std::string, std::string>("cornellbox", "cornellBox"));
	fileLists.push_back(std::pair<std::string, std::string>("crytek_sponza", "crytek_sponza"));
	fileLists.push_back(std::pair<std::string, std::string>("dof-dragons", "dof-dragons"));
	//fileLists.push_back(std::pair<std::string, std::string>("dragonfog", "dragonfog"));
	fileLists.push_back(std::pair<std::string, std::string>("plants-dusk", "plants-dusk_nn"));
	fileLists.push_back(std::pair<std::string, std::string>("poolball", "poolball"));
	fileLists.push_back(std::pair<std::string, std::string>("sanmiguel20", "sanmiguel20"));
	fileLists.push_back(std::pair<std::string, std::string>("sibenik", "sibenik"));
	fileLists.push_back(std::pair<std::string, std::string>("sponzafog", "sponza-fog"));
	fileLists.push_back(std::pair<std::string, std::string>("teapot-metal", "teapot-metal"));
	fileLists.push_back(std::pair<std::string, std::string>("yeahright", "yeahright"));

	//net.TestNN(fileLists);

	const int sampelsNum = 1, max_epochs = 100000;
	const unsigned long long wholeDataSize = 1000 * 1000 * 11 /** 4*/;

	net.BatchTrain(fileLists, sampelsNum, max_epochs, wholeDataSize);

	net.SaveNetwork("Learn.net");


	cudaDeviceReset();

	// Dump memory leaks
	_CrtDumpMemoryLeaks();


	return 0;
}

