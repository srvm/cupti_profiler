# CUPTI CUDA Profiler

This library provides an API for collecting CUDA profiling metrics and events
from within a CUDA application. Programmers specify what metrics and events
they want, and start the profiler before calling a CUDA kernel. The library
sets up the appropriate CUPTI callbacks, calculates the number of
kernel passes required, and gathers values for the specified
metrics and events, and returns them to the programmer.

**Example Usage:**

``` c++
  vector<string> event_names {
                              "active_warps",
                              "gst_inst_32bit",
                              "active_cycles"
                             };
  vector<string> metric_names {
                               "flop_count_dp",
                               "flop_count_sp",
                               "inst_executed"
                              };

  cupti_profiler::profiler profiler(event_names, metric_names);

  // Get #passes required to compute all metrics and events
  const int passes = profiler.get_passes();

  profiler.start();
  for(int i=0; i<passes; ++i) {
    call_kernel(data);
  }
  profiler.stop();

  profiler.print_event_values();
  profiler.print_metric_values();
```
