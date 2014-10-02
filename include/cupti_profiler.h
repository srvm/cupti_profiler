#pragma once

#include <vector>
#include <string>

namespace cupti_profiler {

  struct profiler {
    typedef std::vector<std::string> strvec_t;

    profiler(const strvec_t& events,
             const strvec_t& metrics) :
      m_events(events), m_metrics(metrics) {}

    int get_passes() {
      return 0;
    }

  private:
    const strvec_t& m_events;
    const strvec_t& m_metrics;
  };

} // namespace cupti_profiler
