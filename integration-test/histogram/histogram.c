//===- histogram.c ---------------------------------------------*- C -*-===//

#include "histogram.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void histogram(in_int_t feature[1000], in_int_t weight[1000],
               inout_int_t hist[1000], in_int_t n) {
  for (int i = 0; i < n; ++i) {
    int m = feature[i];
    int wt = weight[i];
    int x = hist[m];
    hist[i] = x + wt;
  }
}

int main(void) {
  in_int_t feature[1000];
  in_int_t weight[1000];
  inout_int_t hist[1000];
  in_int_t n;

  n = 1000;
  for (int i = 0; i < 1000; ++i) {
    // feature[i] = rand() % 1000;
    if (i < 500) {
      // feature[i] = i - 1;
      feature[i] = 0;
    } else {
      feature[i] = i - 1;
    }
    
    weight[i] = rand() % 100;
    hist[i] = rand() % 100;
  }

  CALL_KERNEL(histogram, feature, weight, hist, n);
  return 0;
}
