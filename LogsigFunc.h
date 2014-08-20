#pragma once
#include "ActivationFunc.h"

class LogsigFunc : public ActivationFunc
{
public:
	float activate(const float input) const {
		return (1.f / (1.f + exp(-input) ) );
	}
	float d_activate(const float input) const {
		float s = activate(input);
		return s * (1-s);
	}
};

