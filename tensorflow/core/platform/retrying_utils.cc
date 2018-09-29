/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/retrying_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

namespace {

// In case of failure, every call will be retried kMaxRetries times.
constexpr int kMaxRetries = 10;
// Maximum backoff time in microseconds.
constexpr int64 kMaximumBackoffMicroseconds = 32000000;  // 32 seconds.

bool IsRetriable(const std::set<error::Code> retriable_errors, const error::Code code) {
  return retriable_errors.find(code) != retriable_errors.end();
}

}  // namespace

Status RetryingUtils::CallWithRetries(const std::function<Status()>& f,
                                      const int64 initial_delay_microseconds,
                                      const std::set<error::Code> retriable_errors) {
  return CallWithRetries(f, initial_delay_microseconds, [](int64 micros) {
    return Env::Default()->SleepForMicroseconds(micros);
  }, retriable_errors);
}

Status RetryingUtils::CallWithRetries(
    const std::function<Status()>& f, const int64 initial_delay_microseconds,
    const std::function<void(int64)>& sleep_usec,
    const std::set<error::Code> retriable_errors) {
  int retries = 0;
  while (true) {
    auto status = f();
    if (!IsRetriable(retriable_errors, status.code())) {
      return status;
    }
    if (retries >= kMaxRetries) {
      // Return AbortedError, so that it doesn't get retried again somewhere
      // at a higher level.
      return Status(
          error::ABORTED,
          strings::StrCat(
              "All ", kMaxRetries,
              " retry attempts failed. The last failure: ", status.ToString()));
    }
    int64 delay_micros = 0;
    if (initial_delay_microseconds > 0) {
      const int64 random_micros = random::New64() % 1000000;
      delay_micros = std::min(initial_delay_microseconds << retries,
                              kMaximumBackoffMicroseconds) +
                     random_micros;
    }
    LOG(INFO) << "The operation failed and will be automatically retried in "
              << (delay_micros / 1000000.0) << " seconds (attempt "
              << (retries + 1) << " out of " << kMaxRetries
              << "), caused by: " << status.ToString();
    sleep_usec(delay_micros);
    retries++;
  }
}

Status RetryingUtils::DeleteWithRetries(
    const std::function<Status()>& delete_func,
    const int64 initial_delay_microseconds, 
    const std::set<error::Code> retriable_errors) {
  bool is_retried = false;
  return RetryingUtils::CallWithRetries(
      [delete_func, &is_retried]() {
        const Status status = delete_func();
        if (is_retried && status.code() == error::NOT_FOUND) {
          return Status::OK();
        }
        is_retried = true;
        return status;
      },
      initial_delay_microseconds, retriable_errors);
}

}  // namespace tensorflow
