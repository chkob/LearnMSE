#pragma once
#include <cmath>
#include "Matrix.h"

#define __safe_expf(x) expf((x > SAVE_EXP_THRESHOLD ? SAVE_EXP_THRESHOLD : x))

class ActivationFunc
{
public:
	virtual float activate(const float input) const = 0;
	virtual float d_activate(const float input) const = 0;
};

