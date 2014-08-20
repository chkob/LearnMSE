#pragma once

#include "ActivationFunc.h"
#include <vector>
#include <list>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>

class NeuralNet
{
public:
	NeuralNet(void);
	~NeuralNet(void);

private:
	struct NeuronNode{
		NeuronNode() : model(NULL), bias(0.f), output(0.f), delta(0.f), netin(0.f) {
			lastBiasGrad = 0.f;
			lastBiasDelta = 1.f;
			lastBiasSign = 0;
		}

		ActivationFunc *model;

		// for momentum
		std::vector<float> lastWeightsGrads;
		float lastBiasGrad;

		// for RProp
		std::vector<float> lastWeightDeltas;
		std::vector<int>   lastWeightSigns;
		float lastBiasDelta;
		int   lastBiasSign;

		std::vector<float> weights;
		std::vector<float> gradients;
		float bias, output, delta, netin;
		float ApplyAddBiasActivation(float input) {
			return (output = model->activate((netin=input+bias)));
		}
		float GenNextLayerInput(int nodeID) const {
			return output * weights[nodeID];
		}
		float ComputeDeltaAndGrads(const std::vector<float> &postDeltas) {
			float dActivation = model->d_activate(netin);
			float postSum = 0.f;
			for (int i = 0; i < weights.size(); i++) {
				postSum += weights[i] * postDeltas[i];
				gradients[i] = output * postDeltas[i];
			}
			delta = dActivation * postSum;
			return delta;
		}
		void ComputeGrads(const std::vector<float> &postDeltas) {
			for (int i = 0; i < weights.size(); i++) {
				gradients[i] = output * postDeltas[i];
			}
		}
		void SetOutput(const float out) {
			output = out;	// for firstlayer's output
		}
		void SetNetin(const float netIn) {
			netin = netIn;
		}
		void SetDelta(const float del) {
			delta = del;	// for lastlayer's delta
		}


		float ParApplyAddBiasActivation(float input) const { 
			return model->activate(input + bias);
		}
		float ParGenNextLayerInput(int nodeID, float my_out) const {
			// bug
			//return GenNextLayerInput(nodeID);
			return my_out * weights[nodeID];
		}
		void ParComputeDeltaAndGrads(const float y_out, const float net_in, 
			const std::vector<float> &postDeltas,
			std::vector<float> &grads, float &del,
			const int data_sze = 1) const 
		{
			float postSum = 0.f;	
			for (int i = 0; i < postDeltas.size(); i++) {
				grads[i] = postDeltas[i] * y_out / data_sze;
				postSum += postDeltas[i] * weights[i];
			}
			del = model->d_activate(net_in) * postSum;
		}
		void ParComputeGrads(const float y_out, 
			const std::vector<float> &postDeltas,
			std::vector<float> &grads,
			const int data_sze = 1) const 
		{
			for (int i = 0; i < postDeltas.size(); i++) {
				grads[i] = postDeltas[i] * y_out / data_sze;
			}
		}

	};

	struct Layer{
		std::vector<NeuronNode> neurons;
	};

private:
	float m_learningRate;
	float m_momentum;
	float m_maxGrad, m_minGrad;
	float m_posBonus, m_negBonus;
	std::vector<Layer> m_network;

public:
	void CreateNetwork(int layerNum, int nodeNums[]);
	void SaveNetwork(const std::string &filename) const;
	void LoadNetwork(const std::string &filename);

	void FeedForward(const std::vector<float> &inputvec);
	void BackPropagate(const std::vector<float> &desired_outputvec);
	void UpdateWeights();

	void OnlineTrain(const std::vector<float> &inputvec, const std::vector<float> &desired_outputvec, bool report = false);
	void Run(const std::vector<float> &inputvec, std::vector<float> &outputvec);

	void SetLearningRate(float lr) {	m_learningRate = lr;	}
	void SetMomentum(float m) {		m_momentum = m;		}

	void ParFeedForward(const std::vector<float> &inputvec,
		std::vector<std::vector<float> > &netins, 
		std::vector<std::vector<float> > &y_outs) const;
	struct PackDeltaGrad {
		float delta;
		std::vector<float> grads;
	};
	void ParBackPropagate(const std::vector<float> &desired_outputvec, 
		const std::vector<std::vector<float> > &netins,
		const std::vector<std::vector<float> > &y_outs,
		std::vector<std::vector<PackDeltaGrad> > &packs,
		const int data_sze) const;

	void BatchTrain(const std::vector<std::vector<float> > &inputvecList, 
		const std::vector<std::vector<float> > &outputvecList, 
		const int epochs, const int report_epochs = 500);
};

