//===- histogram.c ---------------------------------------------*- C -*-===//

#include "dynamatic/Integration.h"
#include "histogram_transition.h"
#include <stdlib.h>

void histogram_transition(in_int_t feature[1000],
               inout_int_t hist[1000], in_int_t n) {
  for (int i = 0; i < n; ++i) {

    if (feature[i] > 5) {
      hist[i] = hist[0] + 1;
    }

    hist[i] = hist[0] + 1;
  }
}

int main(void) {
  in_int_t feature[1000];
  inout_int_t hist[1000];
  in_int_t n;

  n = 1000;
  for (int i = 0; i < 1000; ++i) {

    if (i < 500) {
      feature[i] = 4;
    } else {
      feature[i] = 6;
    }

    hist[i] = rand() % 100;
  }

  CALL_KERNEL(histogram_transition, feature, hist, n);
  return 0;
}
