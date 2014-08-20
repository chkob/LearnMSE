#pragma once
#include <cmath>
class ActivationFunc
{
public:
	virtual float activate(const float input) const = 0;
	virtual float d_activate(const float input) const = 0;
};

