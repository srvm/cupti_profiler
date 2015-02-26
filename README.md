# CUDA Profiling Library

This library provides an API for collecting CUDA profiling metrics and events
from within a CUDA application. Programmers specify what metrics and events
they want, and start the profiler before calling one or more CUDA kernels. The library
sets up the appropriate CUPTI callbacks, calculates the number of
kernel passes required, gathers values for the specified
metrics and events, and returns them to the programmer on a per-kernel basis.

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

  printf("Event Trace\n");
  profiler.print_event_values(std::cout);
  printf("Metric Trace\n");
  profiler.print_metric_values(std::cout);
```

**Output:**

```
Event Trace
_Z6kernelIPfEvT_i: (active_warps,1734) (gst_inst_32bit,100) (active_cycles,423) 
_Z7kernel2IPfEvT_i: (active_warps,865) (gst_inst_32bit,50) (active_cycles,418) 

Metric Trace
_Z6kernelIPfEvT_i: (flop_count_dp,0) (flop_count_sp,100) (inst_executed,52) 
_Z7kernel2IPfEvT_i: (flop_count_dp,0) (flop_count_sp,50) (inst_executed,26) 
```
