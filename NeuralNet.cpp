#include "stdafx.h"
#include "NeuralNet.h"
#include "LinearFunc.h"
#include "TansigFunc.h"
#include "LogsigFunc.h"


NeuralNet::NeuralNet(void)
{
	srand(time(NULL));
	m_maxGrad = FLT_MAX;//100000000.f;
	m_minGrad = FLT_MIN;//0.000001f;
	m_posBonus = 1.2f;
	m_negBonus = 0.5f;
	m_learningRate = 0.7f;
	m_momentum = 0.0f;
}

NeuralNet::~NeuralNet(void)
{
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
					neuron.model = new LogsigFunc;
				}
			}
			else
				neuron.model = new LinearFunc;

			neuron.output = 0.f;
			
			layer.neurons.push_back(neuron);
		}
		
		m_network.push_back(layer);
	}

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
				layer.neurons[nodeID].model = new LinearFunc;
		}
		m_network.push_back(layer);
	}

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
							   std::vector<std::vector<float> > &y_outs) const {
//printf("ParFeedForwarding...\n");
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
								 const int data_sze) const {
//printf("ParBackPropagating...\n");
	std::vector<float> postDeltas;
	for (int i = m_network.size()-1; i >= 0; i--) {
		std::vector<PackDeltaGrad> pack;

		if (i == m_network.size()-1) {
			const Layer &lastLayer = m_network[i];
			for (int nodeID = 0; nodeID < lastLayer.neurons.size(); nodeID++) {
				float delta = (desired_outputvec[nodeID] - y_outs[i][nodeID]);
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
				//std::cout << "updateB = " << update << std::endl;
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
					//std::cout << " updateW = " << update << std::endl;
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
	const std::vector<std::vector<float> > &outputvecList, const int epochs, const int report_epochs) {
	const int dataSize = inputvecList.size(), numLayer = m_network.size();

	for (int epoch = 0; epoch < epochs; epoch ++) {
// initial
if (epoch % report_epochs==0)
	printf("Initial...epch %d\n", epoch);		
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
//printf("Accumulating Weights...\n");
		float error = 0.f;
#pragma omp parallel for
		for (int dataID = 0; dataID < dataSize; dataID++) {
			std::vector<std::vector<float> > netins, y_outs;
			ParFeedForward(inputvecList[dataID], netins, y_outs);
			std::vector<std::vector<PackDeltaGrad> > packs;
			ParBackPropagate(outputvecList[dataID], netins, y_outs, packs, dataSize);
			for (int it = 0; it < outputvecList[dataID].size(); it++) {
				error += fabs(outputvecList[dataID][it] - y_outs[numLayer-1][it]);
			}
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

if (epoch % report_epochs==0)
	std::cout << "error = "  << error << std::endl;

// update weights
//printf("Updating Weights...\n");
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
						if (node.lastWeightDeltas.size() == 0) {
							node.lastWeightDeltas.resize(node.weights.size(), 1.f);
							node.lastWeightSigns.resize(node.weights.size(), 0);
						}
						float update = pack.grads[j];
						int sign = SIGN(update);
						int dir = sign * node.lastWeightSigns[j];
						float bonus = dir > 0.f ? m_posBonus : (dir < 0.f ? m_negBonus : 1.f);
						float delta = CLAMP(fabs(node.lastWeightDeltas[j] * bonus), m_minGrad, m_maxGrad);

						node.weights[j] += sign * delta;

						node.lastWeightSigns[j] = sign;
						node.lastWeightDeltas[j] = delta;
					}
				}
			}
		}
if (epoch % report_epochs==0)
	printf("Epoch %d Finished.\n", epoch);
	}

//	printf("Batch Training End.\n");

	return ;
}