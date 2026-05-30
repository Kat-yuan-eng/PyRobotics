#ifndef LOOKUP_TABLE_H
#define LOOKUP_TABLE_H

#include "fixed_point.h"

// === Phase 1: Trig Lookup ===

fp_t lut_sin(int16_t deg);
fp_t lut_cos(int16_t deg);
fp_t lut_atan2(fp_t y, fp_t x);

#endif
