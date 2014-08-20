#pragma once
#include "ActivationFunc.h"

class LinearFunc : public ActivationFunc
{
public:
	float activate(const float input) const {
		return (input);
	}
	float d_activate(const float input) const {
		return 1.f;
	}
};

