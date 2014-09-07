#include "stdafx.h"
#include "NeuralNet.h"
#include "LinearFunc.h"
#include "TansigFunc.h"
#include "LogsigFunc.h"
#include "RectifierFunc.h"
#include <Windows.h>

//#define USE_rmsRprop
//#define USE_Momentum

extern std::string hidden_choice;

NeuralNet::NeuralNet(void)
{
	srand(time(NULL));
	m_maxGrad = 50.f;
	m_minGrad = 1e-6f;
	m_posBonus = 1.2f;
	m_negBonus = 0.5f;
	m_learningRate = 1e-3f;
	m_momentum = 0.7f;
}

NeuralNet::~NeuralNet(void)
{
	delete[] activationFuncsPerLayer;
	delete[] devWeights;
	delete[] devGradients;
	delete[] layerSizes;
}

__forceinline float random01() {
	return (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
}

__forceinline float random(float lower, float upper) {
	return (lower + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(upper-lower))));
}

__forceinline bool NAN(float x) {
	return _isnan(x) || !_finite(x);
}

__forceinline float CLAMP(float x, float m_min, float m_max) {
	return  x < m_min ? m_min : ( x > m_max ? m_max : x );
}

template <typename T> __forceinline int SIGN(T val) {
	return (T(0) < val) - (val < T(0));
}

void NeuralNet::CreateNetwork(int layerNum, int nodeNums[]) {
	for (int i = 0; i < layerNum; i++){
		const int nodeNum = nodeNums[i];
		Layer layer;

		for (int nodeID = 0; nodeID < nodeNum; nodeID++) {

			NeuronNode neuron;

			if (i != 0) { // input layer nodes don't have bias
				neuron.bias = random(-0.5, 0.5);
			}

			if (i != layerNum-1) { // output layer nodes don't have weight (to the next layer), since there is no next layer.
				for (int nextNodeID = 0; nextNodeID < nodeNums[i+1]; nextNodeID++) {
					neuron.weights.push_back(random(-0.5, 0.5));
				}
				neuron.gradients.resize(nodeNums[i+1]);
				if (i != 0){
					neuron.model = new TansigFunc;
				}
			}
			else
				neuron.model = new RectFunc;

			neuron.output = 0.f;

			layer.neurons.push_back(neuron);
		}

		m_network.push_back(layer);
	}

	GPU_Init();

	std::cout << "Nerual network created. " << std::endl;
	return ;
}

void NeuralNet::SaveNetwork(const std::string &filename) const {
	std::ofstream fout(filename.c_str());

	fout << m_network.size() << std::endl;	// layer count

	for (int i = 0; i < m_network.size(); i++) {
		fout << m_network[i].neurons.size() << ' ';	// each layer's neuron count
	}
	fout << std::endl;

	// write each layer's weights & bias
	for (int i = 0; i < m_network.size(); i++) {
		const Layer &layer = m_network[i];
		//fout << "Layer:" << std::endl;
		for (int nodeID = 0; nodeID < layer.neurons.size(); nodeID++) {
			//fout << "Node: " ;
			// write bias
			if (i != 0)
				fout << layer.neurons[nodeID].bias << ' ';

			// write weights
			if (i != m_network.size()-1) {
				for (int j = 0; j < m_network[i+1].neurons.size(); j++) {
					fout << layer.neurons[nodeID].weights[j] << ' ';
				}
			}

			fout << std::endl;
		}

	}

	fout.close();
	std::cout << "Neural network saved. " << std::endl;
	return ;
}

void NeuralNet::LoadNetwork(const std::string &filename) {
	std::ifstream fin(filename.c_str());

	int numLayerSize;
	fin >> numLayerSize;

	std::vector<int> numNeurons(numLayerSize);
	for (int i = 0; i < numLayerSize; i++)
		fin >> numNeurons[i];

	for (int i = 0; i < numLayerSize; i++) {
		Layer layer;
		layer.neurons.resize(numNeurons[i]);
		for (int nodeID = 0; nodeID < numNeurons[i]; nodeID++) {
			if (i != 0)
				fin >> layer.neurons[nodeID].bias;

			if (i != numLayerSize-1) {
				layer.neurons[nodeID].weights.resize(numNeurons[i+1]);
				for (int j = 0; j < numNeurons[i+1]; j++) {
					fin >> layer.neurons[nodeID].weights[j];
				}
				layer.neurons[nodeID].gradients.resize(numNeurons[i+1]);
				if (i != 0){
					layer.neurons[nodeID].model = new TansigFunc;
				}
			}
			else
				layer.neurons[nodeID].model = new RectFunc;
		}
		m_network.push_back(layer);
	}

	GPU_Init();

	return ;
}

void NeuralNet::FeedForward(const std::vector<float> &inputvec) {
	for (int i = 0; i < m_network.size(); i++) {
		Layer &layer = m_network[i];

		for (int nodeID = 0; nodeID < layer.neurons.size(); nodeID++) {
			if (i == 0) {
				layer.neurons[nodeID].SetOutput(inputvec[nodeID]);
				layer.neurons[nodeID].SetNetin(inputvec[nodeID]);
			}
			else {
				float input = 0;	
				const Layer &prevLayer = m_network[i-1];
				for (int prevNodeID = 0; prevNodeID < prevLayer.neurons.size(); prevNodeID++) {
					input += prevLayer.neurons[prevNodeID].GenNextLayerInput(nodeID);
				}
				layer.neurons[nodeID].ApplyAddBiasActivation(input);
			}
		}
	}
	return ;
}

void NeuralNet::ParFeedForward(const std::vector<float> &inputvec, 
							   std::vector<std::vector<float> > &netins,
							   std::vector<std::vector<float> > &y_outs) const 
{
	for (int i = 0; i < m_network.size(); i++) {
		const Layer &layer = m_network[i];
		std::vector<float> netin, y_out;
		for (int nodeID = 0; nodeID < layer.neurons.size(); nodeID++) {
			if (i == 0) {
				netin.push_back(inputvec[nodeID]);
				y_out.push_back(inputvec[nodeID]);
			}
			else {
				float input = 0;	
				const Layer &prevLayer = m_network[i-1];
				for (int prevNodeID = 0; prevNodeID < prevLayer.neurons.size(); prevNodeID++) {
					input += prevLayer.neurons[prevNodeID].ParGenNextLayerInput(nodeID, y_outs[i-1][prevNodeID]);
				}
				netin.push_back(input + layer.neurons[nodeID].bias);
				y_out.push_back(layer.neurons[nodeID].ParApplyAddBiasActivation(input));
			}
		}
		netins.push_back(netin);
		y_outs.push_back(y_out);
	}
	return ;
}

void NeuralNet::Run(const std::vector<float> &inputvec, std::vector<float> &output) {
	FeedForward(inputvec);

	const Layer &lastLayer = m_network[m_network.size()-1];
	for (int i = 0; i < lastLayer.neurons.size(); i++) {
		output.push_back(lastLayer.neurons[i].output);
	}

	return ;
}

void NeuralNet::BackPropagate(const std::vector<float> &desired_outputvec) {
	std::vector<float> postDeltas;
	for (int i = m_network.size()-1; i >= 0; i--) {
		if (i == m_network.size()-1) {
			Layer &lastLayer = m_network[i];
			for (int nodeID = 0; nodeID < lastLayer.neurons.size(); nodeID++) {
				float delta = lastLayer.neurons[nodeID].model->d_activate(lastLayer.neurons[nodeID].netin) *
					(desired_outputvec[nodeID] - lastLayer.neurons[nodeID].output);
				lastLayer.neurons[nodeID].SetDelta(delta);
				postDeltas.push_back(delta);
			}
		}
		else if (i == 0) {
			Layer &firstLayer = m_network[i];
			for (int nodeID = 0; nodeID < firstLayer.neurons.size(); nodeID++) {
				firstLayer.neurons[nodeID].ComputeGrads(postDeltas);
			}
		}
		else {
			Layer &hiddenLayer = m_network[i];
			for (int nodeID = 0; nodeID < hiddenLayer.neurons.size(); nodeID++) {
				hiddenLayer.neurons[nodeID].ComputeDeltaAndGrads(postDeltas);
			}
			postDeltas.clear();
			for (int nodeID = 0; nodeID < hiddenLayer.neurons.size(); nodeID++) {
				postDeltas.push_back(hiddenLayer.neurons[nodeID].delta);
			}
		}
	}
}

void NeuralNet::ParBackPropagate(const std::vector<float> &desired_outputvec, 
								 const std::vector<std::vector<float> > &netins,
								 const std::vector<std::vector<float> > &y_outs,
								 std::vector<std::vector<PackDeltaGrad> > &packs,
								 const int data_sze, float &error) const 
{
	std::vector<float> postDeltas;
	for (int i = m_network.size()-1; i >= 0; i--) {
		std::vector<PackDeltaGrad> pack;

		if (i == m_network.size()-1) {
			const Layer &lastLayer = m_network[i];
			for (int nodeID = 0; nodeID < lastLayer.neurons.size(); nodeID++) {
				float delta = (desired_outputvec[nodeID] - y_outs[i][nodeID]);
				error += fabs(delta);
				postDeltas.push_back(delta);
				PackDeltaGrad packUnit;
				packUnit.delta = delta / data_sze;
				pack.push_back(packUnit);
			}
		}
		else if (i == 0) {
			const Layer &firstLayer = m_network[i];
			for (int nodeID = 0; nodeID < firstLayer.neurons.size(); nodeID++) {
				PackDeltaGrad packUnit;
				packUnit.grads.resize(postDeltas.size());
				firstLayer.neurons[nodeID].ParComputeGrads(
					y_outs[i][nodeID], postDeltas, packUnit.grads, data_sze);
				pack.push_back(packUnit);
			}
		}
		else {
			const Layer &hiddenLayer = m_network[i];
			std::vector<float> newPostDeltas;
			for (int nodeID = 0; nodeID < hiddenLayer.neurons.size(); nodeID++) {
				PackDeltaGrad packUnit;
				packUnit.grads.resize(postDeltas.size());
				hiddenLayer.neurons[nodeID].ParComputeDeltaAndGrads(
					y_outs[i][nodeID], netins[i][nodeID], postDeltas, packUnit.grads, packUnit.delta, data_sze);
				newPostDeltas.push_back(packUnit.delta);
				packUnit.delta /= data_sze;
				pack.push_back(packUnit);
			}
			postDeltas = newPostDeltas;
		}

		packs.push_back(pack);
	}
}

void NeuralNet::UpdateWeights() {
	const int layerNum = m_network.size();
	for (int i = 0; i < layerNum; i++) {
		Layer &layer = m_network[i];
		const int neuronNum = layer.neurons.size();
		for (int nodeID = 0; nodeID < neuronNum; nodeID++) {
			NeuronNode &node = layer.neurons[nodeID];
			if (i != 0) {
				float update = m_learningRate * node.delta + m_momentum * node.lastBiasGrad;
				int sign = SIGN(update);
				update = CLAMP(fabs(update), m_minGrad, m_maxGrad);
				node.lastBiasGrad = sign * update;
				node.bias += sign * update ;
			}

			if (i != layerNum-1) {
				for (int j = 0; j < node.weights.size(); j++) {
					if (node.lastWeightsGrads.size() == 0) {
						node.lastWeightsGrads.resize(node.weights.size(), 0.f);
					}
					float update = m_learningRate * node.gradients[j] + m_momentum * node.lastWeightsGrads[j];
					int sign = SIGN(update);
					update = CLAMP(fabs(update), m_minGrad, m_maxGrad);
					node.lastWeightsGrads[j] = sign * update;
					node.weights[j] += sign * update ;
				}
			}
		}
	}
}

void NeuralNet::OnlineTrain(const std::vector<float> &inputvec, const std::vector<float> &desired_outputvec, bool report) {
	FeedForward(inputvec);
	BackPropagate(desired_outputvec);
	if(report){
		float error = 0.f;
		for (int i = 0; i < desired_outputvec.size(); i++) {
			error += fabs(desired_outputvec[i] - m_network[m_network.size()-1].neurons[i].output);
		}
		std::cout << "error = " << error << std::endl;
	}
	UpdateWeights();
}

void NeuralNet::BatchTrain(const std::vector<std::vector<float> > &inputvecList, 
						   const std::vector<std::vector<float> > &outputvecList, const int epochs, const int report_epochs) 
{
	const int dataSize = inputvecList.size(), numLayer = m_network.size();

	for (int epoch = 0; epoch < epochs; epoch ++) {
		// initial
		if (epoch % report_epochs==0) {
			printf("Initial...epch %d\n", epoch);	
		}

		std::vector<std::vector<PackDeltaGrad> > WeightsForUpdating(numLayer);
		for (int i = 0; i < numLayer; i++) {
			std::vector<PackDeltaGrad> &oneLayerPack = WeightsForUpdating[i];
			for (int nodeID = 0; nodeID < m_network[i].neurons.size(); nodeID++) {
				PackDeltaGrad pack;
				pack.delta = 0.f;
				if (i != numLayer-1) {
					pack.grads.resize(m_network[i+1].neurons.size(), 0.f);
				}
				oneLayerPack.push_back(pack);
			}
		}

		// accumulate weights
		if (epoch % report_epochs==0) {
			printf("Accumulating Weights...\n");
		}

		float error = 0.f;
#pragma omp parallel for
		for (int dataID = 0; dataID < dataSize; dataID++) {
			std::vector<std::vector<float> > netins, y_outs;
			ParFeedForward(inputvecList[dataID], netins, y_outs);
			std::vector<std::vector<PackDeltaGrad> > packs;
			ParBackPropagate(outputvecList[dataID], netins, y_outs, packs, dataSize, error);
#pragma omp critical
			{
				for (int i = 0; i < numLayer; i++) {
					std::vector<PackDeltaGrad> &oneLayerPack = WeightsForUpdating[i];
					std::vector<PackDeltaGrad> &oneLayerTmpPack = packs[numLayer-1-i];
					for (int nodeID = 0; nodeID < m_network[i].neurons.size(); nodeID++) {
						PackDeltaGrad &pack = oneLayerPack[nodeID], &tmpPack = oneLayerTmpPack[nodeID];		
						if (i != 0) {
							pack.delta += tmpPack.delta;
						}
						if (i != numLayer-1) {
							for (int it = 0; it < pack.grads.size(); it++) {
								pack.grads[it] += tmpPack.grads[it];
							}
						}
					}
				}
			}
		}

		if (epoch % report_epochs==0) {
			std::cout << "error = "  << error << std::endl;
		}

		// update weights
		if (epoch % report_epochs==0) {
			printf("Updating Weights...\n");
		}

		for (int i = 0; i < numLayer; i++) {
			Layer &layer = m_network[i];
			std::vector<PackDeltaGrad> &oneLayerPack = WeightsForUpdating[i];
			const int neuronNum = layer.neurons.size();
			for (int nodeID = 0; nodeID < neuronNum; nodeID++) {
				NeuronNode &node = layer.neurons[nodeID];
				PackDeltaGrad &pack = oneLayerPack[nodeID];
				if (i != 0) {
					/*float update = m_learningRate * pack.delta;
					int sign = SIGN(update);
					update = CLAMP(fabs(update), m_minGrad, m_maxGrad);
					node.bias += sign * update;*/
					// RProp			
					float update = pack.delta;
					int sign = SIGN(update);
					int dir = sign * node.lastBiasSign;
					float bonus = dir > 0.f ? m_posBonus : (dir < 0.f ? m_negBonus : 1.f);
					float delta = CLAMP(fabs(node.lastBiasDelta * bonus), m_minGrad, m_maxGrad);

					node.bias += sign * delta;

					node.lastBiasSign = sign;
					node.lastBiasDelta = delta;
				}

				if (i != numLayer-1) {
					for (int j = 0; j < node.weights.size(); j++) {
						/*float update = m_learningRate * pack.grads[j];
						int sign = SIGN(update);
						update = CLAMP(fabs(update), m_minGrad, m_maxGrad);
						node.weights[j] += sign * update;*/
						// RProp	
						if (node.lastWeightDeltas.size() == 0) {
							node.lastWeightDeltas.resize(node.weights.size(), .1f);
							node.lastWeightSigns.resize(node.weights.size(), 0);
						}
						float update = pack.grads[j];
						int sign = SIGN(update);
						int dir = sign * node.lastWeightSigns[j];
						float bonus = dir > 0.f ? m_posBonus : (dir < 0.f ? m_negBonus : .1f);
						float delta = CLAMP(fabs(node.lastWeightDeltas[j] * bonus), m_minGrad, m_maxGrad);

						node.weights[j] += sign * delta;

						node.lastWeightSigns[j] = sign;
						node.lastWeightDeltas[j] = delta;
					}
				}
			}
		}
		SaveNetwork("Learn_MSE.net");

		if (epoch % report_epochs==0) {
			printf("Epoch %d Finished.\n", epoch);
		}

	}

	printf("Batch Training End.\n");

	return ;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

void reshape(std::vector<std::vector<float> > &rst_vec, std::vector<float> &in_vec, const int rows, const int cols) {
	rst_vec.resize(rows);
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		std::vector<float> vec(cols);
		for (int j = 0; j < cols; j++) {
			vec[j] = in_vec[i*cols+j];
		}
		rst_vec[i] = vec;
	}
}

void reshape(std::vector<std::vector<float> > &rst_vec, std::vector<float> &in_vec, const int rows, const int cols, float attach) {
	rst_vec.resize(rows);
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		std::vector<float> vec(cols);
		for (int j = 0; j < cols; j++) {
			vec[j] = in_vec[i*cols+j];
		}
		
		vec.push_back(attach);
		rst_vec[i] = vec;
	}
}

void NeuralNet::ParInitWeightsForUpdating(std::vector<std::vector<PackDeltaGrad> > &WeightsForUpdating) const {
	const int numLayer = m_network.size();
	for (int i = 0; i < numLayer; i++) {
		std::vector<PackDeltaGrad> &oneLayerPack = WeightsForUpdating[i];
		for (int nodeID = 0; nodeID < m_network[i].neurons.size(); nodeID++) {
			PackDeltaGrad pack;
			pack.delta = 0.f;
			if (i != numLayer-1) {
				pack.grads.resize(m_network[i+1].neurons.size(), 0.f);
			}
			oneLayerPack.push_back(pack);
		}
	}
	return ;
}

void NeuralNet::ParUpdateWeights(const std::vector<std::vector<PackDeltaGrad> > &packs, 
								 std::vector<std::vector<PackDeltaGrad> > &WeightsForUpdating,
								 const unsigned long long wholeDataSize) const 
{
	const int numLayer = m_network.size();
	for (int i = 0; i < numLayer; i++) {
		std::vector<PackDeltaGrad> &oneLayerPack = WeightsForUpdating[i];
		const std::vector<PackDeltaGrad> &oneLayerTmpPack = packs[i];
		for (int nodeID = 0; nodeID < m_network[i].neurons.size(); nodeID++) {
			PackDeltaGrad &pack = oneLayerPack[nodeID];
			const PackDeltaGrad &tmpPack = oneLayerTmpPack[nodeID];		
			if (i != 0) {
				pack.delta += tmpPack.delta / wholeDataSize;
			}
			if (i != numLayer-1) {
				for (int it = 0; it < pack.grads.size(); it++) {
					pack.grads[it] += tmpPack.grads[it] / wholeDataSize;
				}
			}
		}
	}
	return ;
}

void NeuralNet::BatchTrain(
	const std::vector<std::pair<std::string, std::string> > &fileLists,
	const int samplesNum, const int epochs, const unsigned long long wholeDataSize) 
{
	const int sceneNum = fileLists.size(), numLayer = m_network.size();
	omp_lock_t locker;
	omp_init_lock(&locker);

	float minError = FLT_MAX;

	openMatlab();
	std::vector<float> errList;

	for (int epoch = 0; epoch < epochs; epoch ++) {
		printf("Epoch %d Begin\n", epoch);

		GPU_CopyWeights();

		std::vector<std::vector<PackDeltaGrad> > WeightsForUpdating(numLayer);
		ParInitWeightsForUpdating(WeightsForUpdating);

		double totalError = 0.f;
		int totalSize = 0;

		for (int sceneID = 0; sceneID < sceneNum; sceneID++) {
		//	int s = rand() % samplesNum + 1;
			const int s = 0;
			double sceneError = 0.f;
			int sceneSize = 0;

			const int sampleRate = 0;
			//for (int sampleRate = 0; sampleRate < 4; sampleRate++) {
				std::string inFile = fileLists[sceneID].first + "\\"+ fileLists[sceneID].second  + "_input" + to_string(s)/*+ to_string(sampleRate) + "_rnd_" + to_string(s) */+".dat";
				std::string outFile = fileLists[sceneID].first + "\\"+ fileLists[sceneID].second + "_output" + to_string(s)/*+ to_string(sampleRate) + "_rnd_" + to_string(s) */+".dat";
				FILE *pFIN, *pFOUT;
				while(!(pFIN = fopen(inFile.c_str(), "rb")))   { 
					printf("Load file = %s, s = %d, sceneID = %d, spRate = %d, infile fail! %s! sleep...\n", inFile.c_str(), 0/*s*/, sceneID, sampleRate, strerror(errno));
					Sleep(1000); 
				}
				while(!(pFOUT = fopen(outFile.c_str(), "rb"))) { 
					printf("Load file = %s, s = %d, sceneID = %d, spRate = %d, outfile fail! %s! sleep...\n", outFile.c_str(), 0/*s*/, sceneID, sampleRate, strerror(errno));
					Sleep(1000); 
				}
				int listSize, perSizeIN, perSizeOUT;
				fread(reinterpret_cast<char*>(&listSize), sizeof(int), 1, pFIN);
				fread(reinterpret_cast<char*>(&listSize), sizeof(int), 1, pFOUT);
				fread(reinterpret_cast<char*>(&perSizeIN), sizeof(int), 1, pFIN);
				fread(reinterpret_cast<char*>(&perSizeOUT), sizeof(int), 1, pFOUT);
				std::vector<float> inBuf, outBuf;
				inBuf.resize(listSize*perSizeIN);
				outBuf.resize(listSize*perSizeOUT);
				fread(reinterpret_cast<char*>(&inBuf[0]), sizeof(std::vector<float>::value_type), listSize*perSizeIN, pFIN);
				fread(reinterpret_cast<char*>(&outBuf[0]), sizeof(std::vector<float>::value_type), listSize*perSizeOUT, pFOUT);

				fclose(pFIN);	fclose(pFOUT);

				std::vector<std::vector<float> > inList, outList;
				
				reshape(inList, inBuf, listSize, perSizeIN, (float) 8*(sampleRate+1) / 32.f);
				reshape(outList, outBuf, listSize, perSizeOUT, (float) 8*(sampleRate+1) / 32.f);

				inBuf.clear();	outBuf.clear();
				
				double error = 0;
				std::vector<std::vector<PackDeltaGrad> > packs(numLayer);
				GPU_MiniBatchTrain(inList, outList, packs, error);
				ParUpdateWeights(packs, WeightsForUpdating, wholeDataSize);
				sceneError += error;
				totalError += error;
				sceneSize += listSize;
				
			//} // End Loop SampleRate
		//}// End Loop random s
			std::cout << "SceneID = " << sceneID << " SceneError = " << sceneError << " " << sceneError / sceneSize << " " << fileLists[sceneID].first <<  " rand s = " << s << " sceneSize = " << sceneSize << std::endl;
			totalSize += sceneSize;
		}// End Loop SceneID


		if (totalError < minError) {
			minError = totalError;
			SaveNetwork("final_all"+hidden_choice+".net");
		}
		std::cout << "Epoch = " << epoch << " MinError = " << minError << " " << minError / totalSize << " TotalError = " << totalError << " " << totalError / totalSize << " totalsize= "<< totalSize << std::endl;
		errList.push_back(totalError/totalSize);
		plotInMatlab((float*)&errList[0], epoch, 0);


		for (int i = 0; i < numLayer; i++) {
			Layer &layer = m_network[i];
			std::vector<PackDeltaGrad> &oneLayerPack = WeightsForUpdating[i];
			const int neuronNum = layer.neurons.size();
			for (int nodeID = 0; nodeID < neuronNum; nodeID++) {
				NeuronNode &node = layer.neurons[nodeID];
				PackDeltaGrad &pack = oneLayerPack[nodeID];
				if (i != 0) {
#ifdef USE_rmsRprop
					/******************************rmsRprop*********************************/
					float biasGrad = pack.delta;
					if (epoch == 0) {
						node.meanSqrBias = biasGrad * biasGrad;
						node.bias += m_learningRate * biasGrad / sqrt(node.meanSqrBias);
#ifdef USE_Momentum
						node.lastDBias = m_learningRate * biasGrad / sqrt(node.meanSqrBias);
#endif
					}
					else{
						node.meanSqrBias = 0.9 * node.meanSqrBias + 0.1 * biasGrad * biasGrad;
#ifdef USE_Momentum
						float dBias = m_learningRate * biasGrad / sqrt(node.meanSqrBias) + m_momentum * node.lastDBias;
						node.bias += dBias;
						node.lastDBias = dBias;
#else
						node.bias += m_learningRate * biasGrad / sqrt(node.meanSqrBias);
#endif
					} 
#else
					/******************************Rprop*********************************/
					float update = pack.delta;
					int sign = SIGN(update);
					int dir = sign * node.lastBiasSign;
					if (dir > EPSILON) {
						float delta = min(fabs(node.lastBiasDelta * m_posBonus), m_maxGrad);
						float dW = sign * delta;
						node.bias += dW * m_learningRate;
						node.lastBiasSign = sign;
						node.lastBiasDelta = delta;
						node.lastDBias = dW;
					}
					else if (dir < -EPSILON) {
						float delta = max(fabs(node.lastBiasDelta * m_negBonus), m_minGrad);
						float dW = -node.lastDBias;
						node.bias += dW * m_learningRate;
						node.lastBiasSign = 0;
						node.lastBiasDelta = delta;
						node.lastDBias = dW;
					}
					else {
						float delta = node.lastBiasDelta;
						float dW = sign * delta;
						node.bias += dW * m_learningRate;
						node.lastBiasSign = sign;
						node.lastBiasDelta = delta;
						node.lastDBias = dW;
					}
#endif
				}
				if (i != numLayer-1) {
					for (int j = 0; j < node.weights.size(); j++) {
#ifdef USE_rmsRprop
						/******************************rmsRprop*********************************/
						if (node.meanSqrGrads.size() == 0) {
							node.meanSqrGrads.resize(node.weights.size(), 0.f);	
							node.lastDWeights.resize(node.weights.size(), 0.f);
						}
						float weightGrad = pack.grads[j];
						if (epoch == 0) {
							node.meanSqrGrads[j] = weightGrad * weightGrad;
							node.weights[j] += m_learningRate * weightGrad / sqrt(node.meanSqrGrads[j]);
#ifdef USE_Momentum
							node.lastDWeights[j] = m_learningRate * weightGrad / sqrt(node.meanSqrGrads[j]);
#endif
						}
						else {
							node.meanSqrGrads[j] = 0.9 * node.meanSqrGrads[j] + 0.1 * weightGrad * weightGrad;
#ifdef USE_Momentum
							float dWeight = m_learningRate * weightGrad / sqrt(node.meanSqrGrads[j]) + m_momentum * node.lastDWeights[j];
							node.weights[j] += dWeight;
							node.lastDWeights[j] = dWeight;
#else
							node.weights[j] += m_learningRate * weightGrad / sqrt(node.meanSqrGrads[j]);
#endif
						}
#else
						/******************************Rprop*********************************/
						if (node.lastWeightDeltas.size() == 0) {
							node.lastWeightDeltas.resize(node.weights.size(), 0.1f);
							node.lastWeightSigns.resize(node.weights.size(), 0);
							node.lastDWeights.resize(node.weights.size(), 0.f);
						}
						float update = pack.grads[j];
						int sign = SIGN(update);
						int dir = sign * node.lastWeightSigns[j];
						if (dir > EPSILON) {
							float delta = min(fabs(node.lastWeightDeltas[j] * m_posBonus), m_maxGrad);
							float dW = sign * delta;
							node.weights[j] += dW * m_learningRate;
							node.lastWeightSigns[j] = sign;
							node.lastWeightDeltas[j] = delta;
							node.lastDWeights[j] = dW;
						}
						else if (dir < -EPSILON) {
							float delta = max(fabs(node.lastWeightDeltas[j] * m_negBonus), m_minGrad);
							float dW = -node.lastDWeights[j];
							node.weights[j] += dW * m_learningRate;
							node.lastWeightSigns[j] = 0;
							node.lastWeightDeltas[j] = delta;
							node.lastDWeights[j] = dW;
						}
						else {
							float delta = node.lastWeightDeltas[j];
							float dW = sign * delta;
							node.weights[j] += dW * m_learningRate;
							node.lastWeightSigns[j] = sign;
							node.lastWeightDeltas[j] = delta;
							node.lastDWeights[j] = dW;
						}
#endif
					}
				}
			}
		}

		printf("Epoch %d Finished!\n", epoch);

		SaveNetwork("Learn_all"+hidden_choice+".net");

	}// End Loop Epoch

	closeMatlab();

	return ;
}


void NeuralNet::MiniBatchTrain(
	const std::vector<std::vector<float> > &inList,
	const std::vector<std::vector<float> > &outList,
	std::vector<std::vector<PackDeltaGrad> > &thepacks) const 
{
	clock_t tic = clock();
	const int numLayer = m_network.size();
	for (int i = 0; i < numLayer; i++) {
		std::vector<PackDeltaGrad> &oneLayerPack = thepacks[i];
		for (int nodeID = 0; nodeID < m_network[i].neurons.size(); nodeID++) {
			PackDeltaGrad pack;
			pack.delta = 0.f;
			if (i != numLayer-1) {
				pack.grads.resize(m_network[i+1].neurons.size(), 0.f);
			}
			oneLayerPack.push_back(pack);
		}
	}
	float error = 0.f;

#pragma omp parallel for 
	for (int dataID = 0; dataID < inList.size(); dataID++) {
		std::vector<std::vector<float> > netins, y_outs;
		ParFeedForward(inList[dataID], netins, y_outs);
		std::vector<std::vector<PackDeltaGrad> > packs;
		ParBackPropagate(outList[dataID], netins, y_outs, packs, 1.f, error);
		for (int i = 0; i < numLayer; i++) {
			std::vector<PackDeltaGrad> &oneLayerPack = thepacks[i];
			std::vector<PackDeltaGrad> &oneLayerTmpPack = packs[numLayer-1-i];
			for (int nodeID = 0; nodeID < m_network[i].neurons.size(); nodeID++) {
				PackDeltaGrad &pack = oneLayerPack[nodeID], &tmpPack = oneLayerTmpPack[nodeID];		
#pragma omp critical 
				{	
					if (i != 0) {
						pack.delta += tmpPack.delta;
					}
					if (i != numLayer-1) {
						for (int it = 0; it < pack.grads.size(); it++) {
							pack.grads[it] += tmpPack.grads[it];
						}
					}
				}

			}
		}
	}


	printf("Mini-Batch training, timeuse = %d msecs, error = %f\n", clock()-tic, error);
}

void AddBiasApplyActivation(FUNC func, Matrix<float>& A, Matrix<float>& B);
void SubtractBFromA(Matrix<float>& A, Matrix<float>& B, Matrix<float>& C);
void SubtractBFromARelative(Matrix<float>& A, Matrix<float>& B, Matrix<float>& C);
void SubtractBFromARelative2(Matrix<float>& A, Matrix<float>& B, Matrix<float>& C);
void MultApplyActivationDeriv(FUNC func, Matrix<float>& A, Matrix<float>& B);
void RProp(Matrix<float>& currentWeights, Matrix<float>& currentGradients, Matrix<float>& meanSquaredGrad, float learningRate, int iteration);
void AddAToB(Matrix<float>& A, Matrix<float>& B);
void MultAByb(Matrix<float>& A, float b);

void NeuralNet::GPU_Init() {
	const int numOfLayers = m_network.size();
	layerSizes = new int[numOfLayers];
	for (int i = 0; i < numOfLayers; i++) {
		layerSizes[i] = m_network[i].neurons.size();
	}
	activationFuncsPerLayer = new FUNC[2*(numOfLayers-1)];
	for (int i = 0; i < numOfLayers - 1; i++) {
		if (i == numOfLayers-2) {
			activationFuncsPerLayer[2*i] = REC_LINEAR;
			activationFuncsPerLayer[2*i + 1] = REC_LINEAR_DERIV;
		}
		else {
			activationFuncsPerLayer[2*i] = TAN_SIG;
			activationFuncsPerLayer[2*i + 1] = TAN_SIG_DERIV;
		}
	}
	devWeights = new Matrix<float>[2*(numOfLayers-1)];
	devGradients = new Matrix<float>[2*(numOfLayers-1)];

	CublasErrorCheck(cublasCreate(&handle));
}

void NeuralNet::GPU_CopyWeights() const {
	const int numOfLayers = m_network.size();

	for(int i = 0; i < numOfLayers-1; i++) 
	{
		// Weights
		Matrix<float> tmpWeights(layerSizes[i], layerSizes[i + 1]);
		tmpWeights.AllocateData(false);


		devWeights[2*i] = Matrix<float>(layerSizes[i], layerSizes[i + 1]);
		devWeights[2*i].AllocateData(true);

		//tmpWeights.initializeToRandom(-rndBound, rndBound);
		const Layer &layer = m_network[i];
		for (int col = 0; col < layerSizes[i]; col++) {
			const NeuronNode &node = layer.neurons[col];
			for (int row = 0; row < layerSizes[i+1]; row++) {
				tmpWeights.setElement(row, col, node.weights[row]);
			}
		}
		tmpWeights.HostToDevice(devWeights[2*i]);

		// Bias
		Matrix<float> tmpBiases(1, layerSizes[i + 1]);
		tmpBiases.AllocateData(false);
		devWeights[2*i+1] = Matrix<float>(1, layerSizes[i + 1]);
		devWeights[2*i+1].AllocateData(true);

		/*tmpBiases.initializeToRandom(-rndBound, rndBound);*/
		const Layer &nextlayer = m_network[i+1];
		for (int row = 0; row < layerSizes[i+1]; row++) {
			tmpBiases.setElement(row, nextlayer.neurons[row].bias);
		}
		tmpBiases.HostToDevice(devWeights[2*i+1]);

		// Gradient for weights
		devGradients[2*i] = Matrix<float>(layerSizes[i], layerSizes[i + 1]);
		devGradients[2*i].AllocateData(true);
		devGradients[2*i].SetToZero();
		// Gradients for biases
		devGradients[2*i+1] = Matrix<float>(1, layerSizes[i + 1]);
		devGradients[2*i+1].AllocateData(true);
		devGradients[2*i+1].SetToZero();
	}
}

void NeuralNet::GPU_FeedForward(Matrix<float> &input, Matrix<float> *activations) {
	activations[0] = Matrix<float>(input.getWidth(), input.getHeight());
	activations[0].AllocateData(true);	
	input.HostToDevice(activations[0]);
	const int numOfLayers = m_network.size();
	for (int i = 0; i < numOfLayers - 1; i++)
	{
		activations[i+1] = Matrix<float> (activations[i].getWidth(), devWeights[2*i].getHeight());
		activations[i+1].AllocateData(true);

		MatMultAB(handle, devWeights[2*i], activations[i], activations[i+1]);

		AddBiasApplyActivation(activationFuncsPerLayer[2*i], activations[i+1], devWeights[2*i+1]);

		if (i < numOfLayers - 1)
			activations[i].DeviceToHost();

	}
}

void NeuralNet::GPU_BackPropagation(Matrix<float> *gradWeights, Matrix<float> *delta, Matrix<float> *activations) {
	const int numOfLayers = m_network.size();
	for (int i = numOfLayers - 3; i >= 0; i--)
	{
		delta[i] = Matrix<float> (delta[i+1].getWidth(), devWeights[2*(i+1)].getWidth());
		delta[i].AllocateData(true);

		MatMultATB(handle, devWeights[2*(i+1)], delta[i+1], delta[i]);

		activations[i+1].HostToDevice();

		MultApplyActivationDeriv(activationFuncsPerLayer[2*i+1], delta[i], activations[i+1]);

		MatMultABT(handle, delta[i+1], activations[i+1], gradWeights[2*(i+1)]);
		activations[i+1].~Matrix();

		SumRows(handle, delta[i+1], gradWeights[2*(i+1)+1]);
		delta[i+1].~Matrix();

	}
	activations[0].HostToDevice();
	MatMultABT(handle, delta[0], activations[0], gradWeights[0]);
	activations[0].~Matrix();
	SumRows(handle, delta[0], gradWeights[1]);
	delta[0].~Matrix();
}

void NeuralNet::ComputeOutputDelta(Matrix<float>& activation, Matrix<float>& outputGT, Matrix<float>& delta, double &error) 
{
	const int numOfLayers = m_network.size();

	delta = Matrix<float>(activation.getWidth(), activation.getHeight());
	delta.AllocateData(true);
	SubtractBFromA(activation, outputGT, delta);
	outputGT.~Matrix();
	float err = 0;
	ComputeNorm2(handle, delta, err);
	error = err;
	error *= error;

	MultApplyActivationDeriv(activationFuncsPerLayer[2*(numOfLayers-2)+1], delta, activation);
	
	activation.DeviceToHost(); // We take it to the host to write it to a file at the end.
}

void NeuralNet::GPU_MiniBatchTrain(
	const std::vector<std::vector<float> > &inList,
	const std::vector<std::vector<float> >& outList,
	std::vector<std::vector<PackDeltaGrad> > &thepacks,
	double &error)  
{
	clock_t tic = clock();

	const int numOfLayers = m_network.size();
	const int numInputs = m_network[0].neurons.size();
	const int numOutputs = m_network[numOfLayers-1].neurons.size();
	Matrix<float> input = Matrix<float> (inList.size(), numInputs);
	Matrix<float> desired_output = Matrix<float> (outList.size(), numOutputs);
	
	input.AllocateData(false);
	desired_output.AllocateData(false);
	for (int dataID = 0; dataID < inList.size(); dataID++) {
		Matrix<float> inMat(1, numInputs), outMat(1, numOutputs);
		inMat.AllocateData(false);
		outMat.AllocateData(false);
		for (int e = 0; e < numInputs; e++) {
			inMat.setElement(e, inList[dataID][e]);
		}
		for (int e = 0; e < numOutputs; e++) {
			outMat.setElement(e, outList[dataID][e]);
		}
		input.Insert(inMat, dataID);
		desired_output.Insert(outMat, dataID);
	}

	desired_output.HostToDevice();

	Matrix<float>* delta = new Matrix<float>[numOfLayers-1];
	Matrix<float>* activations = new Matrix<float>[numOfLayers];
	
	GPU_FeedForward(input, activations);
	
	ComputeOutputDelta(activations[numOfLayers-1], desired_output, delta[numOfLayers-2], error);
	
	GPU_BackPropagation(devGradients, delta, activations);

	delete[] activations;
	delete[] delta;
	
	// TODO :  GET BACK MY GRADIENTS!
	thepacks.resize(numOfLayers);
	ParInitWeightsForUpdating(thepacks);


	for (int i = 0; i < numOfLayers - 1; i ++) {
		// Gradient for weights
		devGradients[2*i].DeviceToHost();
		Matrix<float> &devWeights = devGradients[2*i];
		/*devGradients[2*i] = Matrix<float>(layerSizes[i], layerSizes[i + 1]);
		devGradients[2*i].AllocateData(true);
		devGradients[2*i].SetToZero();*/
		std::vector<PackDeltaGrad> &thisLayer = thepacks[i];
		std::vector<PackDeltaGrad> &nextLayer = thepacks[i+1];
		for (int nodeID = 0; nodeID < thisLayer.size(); nodeID++) {
			for (int nextNodeID = 0; nextNodeID < nextLayer.size(); nextNodeID++) {
				thisLayer[nodeID].grads[nextNodeID] = -devWeights.getElement(nodeID, nextNodeID);
			}
		}
		devGradients[2*i].HostToDevice();

		// Gradients for biases
		devGradients[2*i+1].DeviceToHost();
		Matrix<float> &devBiases = devGradients[2*i+1];
		/*devGradients[2*i+1] = Matrix<float>(1, layerSizes[i + 1]);
		devGradients[2*i+1].AllocateData(true);
		devGradients[2*i+1].SetToZero();*/
		for (int nextNodeID = 0; nextNodeID < nextLayer.size(); nextNodeID++) {
			nextLayer[nextNodeID].delta = -devBiases.getElement(nextNodeID);
		}
		devGradients[2*i+1].HostToDevice();

	}

	//printf("GPU Mini-Batch training, timeuse = %d msecs, error = %f\n", clock()-tic, error);
}
 
void NeuralNet::TestNN(const std::vector<std::pair<std::string, std::string> > &fileLists) {
	for (int sceneID = 0; sceneID < fileLists.size(); sceneID++) {
		for (int sampleRate = 0; sampleRate < 4; sampleRate ++) {
			const int s = rand() % 5 + 1;

			std::string inF = fileLists[sceneID].first + "\\"/* + std::to_string(s) + "\\" */+ fileLists[sceneID].second  + "_input"+ to_string(sampleRate) + "_rnd_" + to_string(s) +".dat";
			std::string outF = fileLists[sceneID].first + "\\"/* + std::to_string(s) + "\\" */+ fileLists[sceneID].second + "_output"+ to_string(sampleRate) + "_rnd_" + to_string(s) +".dat";
			std::cout << "INF = " << inF << " outF = " << outF <<  " s = " << s  << " spRate = " << sampleRate << std::endl;
			FILE *pFIN, *pFOUT;
			pFIN = fopen(inF.c_str(), "rb");
			pFOUT = fopen(outF.c_str(), "rb");
			int listSize, perSizeIN, perSizeOUT;
			fread(reinterpret_cast<char*>(&listSize), sizeof(int), 1, pFIN);
			fread(reinterpret_cast<char*>(&listSize), sizeof(int), 1, pFOUT);
			fread(reinterpret_cast<char*>(&perSizeIN), sizeof(int), 1, pFIN);
			fread(reinterpret_cast<char*>(&perSizeOUT), sizeof(int), 1, pFOUT);
			std::vector<float> inBuf, outBuf;
			inBuf.resize(listSize*perSizeIN);
			outBuf.resize(listSize*perSizeOUT);
			fread(reinterpret_cast<char*>(&inBuf[0]), sizeof(std::vector<float>::value_type), listSize*perSizeIN, pFIN);
			fread(reinterpret_cast<char*>(&outBuf[0]), sizeof(std::vector<float>::value_type), listSize*perSizeOUT, pFOUT);

			fclose(pFIN);	fclose(pFOUT);

			std::vector<std::vector<float> > inList, outList;
			reshape(inList, inBuf, listSize, perSizeIN, (float)(sampleRate+1)*8/32.f);
			reshape(outList, outBuf, listSize, perSizeOUT, (float)(sampleRate+1)*8/32.f);
			inBuf.clear();	outBuf.clear();

			std::ofstream fout2(fileLists[sceneID].first + "_spRate_" + to_string(sampleRate) + "_nnOut.txt");

			for (int i = 0; i < listSize; i++) {
				std::vector<float> in = inList[i], out = outList[i], nnOut;
				this->Run(in, nnOut);
				fout2 << nnOut[0] << " " << out[0] << std::endl;
			}

			fout2.close();
		}
	}
	return ;
}


void NeuralNet::openMatlab() {
	assert(MatlabPloter == NULL);

	if (!(MatlabPloter = engOpen(""))) {
		fprintf(stderr, "ERROR:Can't start MATLAB engine!\n");
		return;
	}
}

void NeuralNet::plotInMatlab(float *errList, int curEpoch, int startEpoch) {
	// Matlab array
	mxArray* error = NULL;

	int errorSize = curEpoch - startEpoch + 1;

	// Create a vector in matlab and copy the data over
	error = mxCreateDoubleMatrix(1, errorSize, mxREAL);
	
	for(int i = startEpoch; i <= curEpoch; i++) {
		double* errorPtr = (double *) mxGetPr(error);
		errorPtr[i-startEpoch] = errList[i];
	}

	// Put the variable in the environment and plot
	engPutVariable(MatlabPloter, "error", error);
	engEvalString(MatlabPloter, "plot(error);");
	engEvalString(MatlabPloter, "title('Error Plot');");
	engEvalString(MatlabPloter, "drawnow;");

	// Delete matlab array
	mxDestroyArray(error);
	
}

void NeuralNet::closeMatlab() {
	if (MatlabPloter) {
		engEvalString(MatlabPloter, "close");
		engClose(MatlabPloter);
	}
}