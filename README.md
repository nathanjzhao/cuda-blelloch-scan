# CUDA Blelloch Scan

Work-efficient implmentation of Blelloch scan for finding prefix sum with parallelized multi-threaded algorithm in $O(\log{N})$ steps.

GPU version is slower at the moment than CPU as the memory transfers are expensive and much alrger N is necessary to make the optimizations worthwhile.