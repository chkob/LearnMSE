#pragma once
#include "ActivationFunc.h"

class TansigFunc : public ActivationFunc
{
public:
	float activate(const float input) const {
		return (__safe_expf(input) - __safe_expf(-input)) / (__safe_expf(input) + __safe_expf(-input));//(2 / (1 + exp(-2*input)) - 1);
	}

	float d_activate(const float input) const {
		float s = activate(input);
		return (1 - s*s);
	}
};

