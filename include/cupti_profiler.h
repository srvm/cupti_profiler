#pragma once

#include <vector>
#include <string>
#include <cupti.h>

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("%s:%d: error %d for CUDA Driver API function '%s'\n",    \
              __FILE__, __LINE__, err, cufunc);                         \
      exit(-1);                                                         \
    }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
  if (err != CUPTI_SUCCESS)                                             \
    {                                                                   \
      const char *errstr;                                               \
      cuptiGetResultString(err, &errstr);                               \
      printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",          \
              __FILE__, __LINE__, errstr, cuptifunc);                   \
      exit(-1);                                                         \
    }

namespace cupti_profiler {
namespace detail {

  struct cupti_event_data {
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
  };

  // Structure to hold data collected by callback
  struct runtime_api_trace {
    cupti_event_data *eventData;
    uint64_t eventVal;
  };

  void CUPTIAPI
  __get_event_value_callback(void *userdata,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid,
                             const CUpti_CallbackData *cbInfo) {
    CUptiResult cuptiErr;
    detail::runtime_api_trace *traceData =
      (detail::runtime_api_trace*)userdata;
    size_t bytesRead;

    // This callback is enabled only for launch so we shouldn't see anything else.
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
      printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
      exit(-1);
    }

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      cudaDeviceSynchronize();
      cuptiErr = cuptiSetEventCollectionMode(cbInfo->context,
                                             CUPTI_EVENT_COLLECTION_MODE_KERNEL);
      CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");
      cuptiErr = cuptiEventGroupEnable(traceData->eventData->eventGroup);
      CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");
    }

    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      bytesRead = sizeof (uint64_t);
      cudaDeviceSynchronize();
      cuptiErr = cuptiEventGroupReadEvent(traceData->eventData->eventGroup,
                                          CUPTI_EVENT_READ_FLAG_NONE,
                                          traceData->eventData->eventId,
                                          &bytesRead, &traceData->eventVal);
      CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");

      cuptiErr = cuptiEventGroupDisable(traceData->eventData->eventGroup);
      CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");
    }
  }

} // namespace detail

  struct profiler {
    typedef std::vector<std::string> strvec_t;
    typedef std::vector<double> val_t;

    profiler(const strvec_t& events,
             const strvec_t& metrics,
             const int device_num=0) :
      m_event_names(events),
      m_metric_names(metrics),
      m_device_num(0) {

      m_err = cuInit(0);
      CHECK_CU_ERROR(m_err, "cuInit");

      int device_count = 0;
      m_err = cuDeviceGetCount(&device_count);
      CHECK_CU_ERROR(m_err, "cuDeviceGetCount");

      if (device_count == 0) {
        printf("[Error]: There is no device supporting CUDA.\n");
        exit(1);
      }

      m_err = cuDeviceGet(&m_device, device_num);
      CHECK_CU_ERROR(m_err, "cuDeviceGet");

      m_err = cuCtxCreate(&m_context, 0, m_device);
      CHECK_CU_ERROR(m_err, "cuCtxCreate");

      m_cupti_err = cuptiEventGroupCreate(m_context,
                    &m_cupti_event.eventGroup, 0);
      CHECK_CUPTI_ERROR(m_cupti_err, "cuptiEventGroupCreate");

      m_cupti_err = cuptiEventGetIdFromName(m_device,
                    m_event_names[0].c_str(),
                    &m_cupti_event.eventId);
      if(m_cupti_err != CUPTI_SUCCESS) {
        printf("Invalid event name: %s\n", m_event_names[0].c_str());
        exit(1);
      }
    }

    ~profiler() {
    }

    int get_passes() {
      return 0;
    }

    void start() {
      m_cupti_err = cuptiEventGroupAddEvent(m_cupti_event.eventGroup,
                    m_cupti_event.eventId);
      CHECK_CUPTI_ERROR(m_cupti_err, "cuptiEventGroupAddEvent");

      m_trace.eventData = &m_cupti_event;

      m_cupti_err = cuptiSubscribe(&m_subscriber,
          (CUpti_CallbackFunc)detail::__get_event_value_callback , &m_trace);
      CHECK_CUPTI_ERROR(m_cupti_err, "cuptiSubscribe");

      m_cupti_err = cuptiEnableCallback(1, m_subscriber,
          CUPTI_CB_DOMAIN_RUNTIME_API,
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
      CHECK_CUPTI_ERROR(m_cupti_err, "cuptiEnableCallback");
    }

    void stop() {
      m_events.push_back(m_trace.eventVal);

      m_trace.eventData = NULL;

      m_cupti_err = cuptiEventGroupRemoveEvent(m_cupti_event.eventGroup,
                    m_cupti_event.eventId);
      CHECK_CUPTI_ERROR(m_cupti_err, "cuptiEventGroupRemoveEvent");

      m_cupti_err = cuptiEventGroupDestroy(m_cupti_event.eventGroup);
      CHECK_CUPTI_ERROR(m_cupti_err, "cuptiEventGroupDestroy");

      m_cupti_err = cuptiUnsubscribe(m_subscriber);
      CHECK_CUPTI_ERROR(m_cupti_err, "cuptiUnsubscribe");
    }

    void print_event_values() {
      printf("event_name: %s, value = %lf\n",
             m_event_names[0].c_str(), m_events[0]);
    }

    val_t get_event_values()
    { return m_events; }

    val_t get_metric_values()
    { return m_metrics; }

  private:

  private:
    int m_device_num;
    const strvec_t& m_event_names;
    const strvec_t& m_metric_names;
    val_t m_events;
    val_t m_metrics;

    CUcontext m_context;
    CUdevice m_device;
    CUresult m_err;

    CUptiResult m_cupti_err;
    CUpti_SubscriberHandle m_subscriber;
    detail::cupti_event_data m_cupti_event;
    detail::runtime_api_trace m_trace;
  };

} // namespace cupti_profiler
