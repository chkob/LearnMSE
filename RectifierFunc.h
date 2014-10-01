#pragma once
#include "ActivationFunc.h"

class RectFunc : public ActivationFunc
{
public:
	float activate(const float input) const {
		return ( logf(1.0 + __safe_expf(input)) );
	}

	float d_activate(const float input) const {
		return ( 1.0 / (1.0 + __safe_expf(-input)) );
	}
};

