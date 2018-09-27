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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_FILE_SYSTEM_H_

#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/retrying_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

/// A wrapper to add retry logic to another file system.
template <typename Underlying>
class RetryingFileSystem : public FileSystem {
 public:
  RetryingFileSystem(std::unique_ptr<Underlying> base_file_system,
                     std::set<error::Code> retriable_errors = {error::UNAVAILABLE, error::DEADLINE_EXCEEDED, error::UNKNOWN},
                     int64 delay_microseconds = 1000000)
      : base_file_system_(std::move(base_file_system)),
        retriable_errors_(retriable_errors),
        initial_delay_microseconds_(delay_microseconds)
         {}

  // RetryingFileSystem(std::unique_ptr<Underlying> base_file_system,
  //                    int64 delay_microseconds = 1000000)
  //     : base_file_system_(std::move(base_file_system)),
  //       initial_delay_microseconds_(delay_microseconds) {
  //         retriable_errors_ = ;
  //       }

  Status NewRandomAccessFile(
      const string& filename,
      std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const string& fname) override {
    return RetryingUtils::CallWithRetries(
        [this, &fname]() { return base_file_system_->FileExists(fname); },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    return RetryingUtils::CallWithRetries(
        [this, &dir, result]() {
          return base_file_system_->GetChildren(dir, result);
        },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* result) override {
    return RetryingUtils::CallWithRetries(
        [this, &pattern, result]() {
          return base_file_system_->GetMatchingPaths(pattern, result);
        },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status Stat(const string& fname, FileStatistics* stat) override {
    return RetryingUtils::CallWithRetries(
        [this, &fname, stat]() { return base_file_system_->Stat(fname, stat); },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status DeleteFile(const string& fname) override {
    return RetryingUtils::DeleteWithRetries(
        [this, &fname]() { return base_file_system_->DeleteFile(fname); },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status CreateDir(const string& dirname) override {
    return RetryingUtils::CallWithRetries(
        [this, &dirname]() { return base_file_system_->CreateDir(dirname); },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status DeleteDir(const string& dirname) override {
    return RetryingUtils::DeleteWithRetries(
        [this, &dirname]() { return base_file_system_->DeleteDir(dirname); },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    return RetryingUtils::CallWithRetries(
        [this, &fname, file_size]() {
          return base_file_system_->GetFileSize(fname, file_size);
        },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status RenameFile(const string& src, const string& target) override {
    return RetryingUtils::CallWithRetries(
        [this, &src, &target]() {
          return base_file_system_->RenameFile(src, target);
        },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status IsDirectory(const string& dirname) override {
    return RetryingUtils::CallWithRetries(
        [this, &dirname]() { return base_file_system_->IsDirectory(dirname); },
        initial_delay_microseconds_, retriable_errors_);
  }

  Status DeleteRecursively(const string& dirname, int64* undeleted_files,
                           int64* undeleted_dirs) override {
    return RetryingUtils::DeleteWithRetries(
        [this, &dirname, undeleted_files, undeleted_dirs]() {
          return base_file_system_->DeleteRecursively(dirname, undeleted_files,
                                                      undeleted_dirs);
        },
        initial_delay_microseconds_, retriable_errors_);
  }

  void FlushCaches() override { base_file_system_->FlushCaches(); }

  Underlying* underlying() const { return base_file_system_.get(); }

 private:
  std::unique_ptr<Underlying> base_file_system_;
  const int64 initial_delay_microseconds_;
  const std::set<error::Code> retriable_errors_;
  TF_DISALLOW_COPY_AND_ASSIGN(RetryingFileSystem);
};

namespace retrying_internals {

class RetryingRandomAccessFile : public RandomAccessFile {
 public:
  RetryingRandomAccessFile(std::unique_ptr<RandomAccessFile> base_file,
                           int64 delay_microseconds, 
                           std::set<error::Code> retriable_errors)
      : base_file_(std::move(base_file)),
        initial_delay_microseconds_(delay_microseconds),
        retriable_errors_(retriable_errors) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    return RetryingUtils::CallWithRetries(
        [this, offset, n, result, scratch]() {
          return base_file_->Read(offset, n, result, scratch);
        },
        initial_delay_microseconds_, retriable_errors_);
  }

 private:
  std::unique_ptr<RandomAccessFile> base_file_;
  const int64 initial_delay_microseconds_;
  const std::set<error::Code> retriable_errors_;
};

class RetryingWritableFile : public WritableFile {
 public:
  RetryingWritableFile(std::unique_ptr<WritableFile> base_file,
                       int64 delay_microseconds,
                       std::set<error::Code> retriable_errors)
      : base_file_(std::move(base_file)),
        initial_delay_microseconds_(delay_microseconds),
        retriable_errors_(retriable_errors) {}

  ~RetryingWritableFile() override {
    // Makes sure the retrying version of Close() is called in the destructor.
    Close().IgnoreError();
  }

  Status Append(StringPiece data) override {
    return RetryingUtils::CallWithRetries(
        [this, &data]() { return base_file_->Append(data); },
        initial_delay_microseconds_, retriable_errors_);
  }
  Status Close() override {
    return RetryingUtils::CallWithRetries(
        [this]() { return base_file_->Close(); }, initial_delay_microseconds_, retriable_errors_);
  }
  Status Flush() override {
    return RetryingUtils::CallWithRetries(
        [this]() { return base_file_->Flush(); }, initial_delay_microseconds_, retriable_errors_);
  }
  Status Sync() override {
    return RetryingUtils::CallWithRetries(
        [this]() { return base_file_->Sync(); }, initial_delay_microseconds_, retriable_errors_);
  }

 private:
  std::unique_ptr<WritableFile> base_file_;
  const int64 initial_delay_microseconds_;
  const std::set<error::Code> retriable_errors_;
};

}  // namespace retrying_internals

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewRandomAccessFile(
    const string& filename, std::unique_ptr<RandomAccessFile>* result) {
  std::unique_ptr<RandomAccessFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      [this, &filename, &base_file]() {
        return base_file_system_->NewRandomAccessFile(filename, &base_file);
      },
      initial_delay_microseconds_, retriable_errors_));
  result->reset(new retrying_internals::RetryingRandomAccessFile(
      std::move(base_file), initial_delay_microseconds_, retriable_errors_));
  return Status::OK();
}

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewWritableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      [this, &filename, &base_file]() {
        return base_file_system_->NewWritableFile(filename, &base_file);
      },
      initial_delay_microseconds_, retriable_errors_));
  result->reset(new retrying_internals::RetryingWritableFile(
      std::move(base_file), initial_delay_microseconds_, retriable_errors_));
  return Status::OK();
}

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewAppendableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      [this, &filename, &base_file]() {
        return base_file_system_->NewAppendableFile(filename, &base_file);
      },
      initial_delay_microseconds_, retriable_errors_));
  result->reset(new retrying_internals::RetryingWritableFile(
      std::move(base_file), initial_delay_microseconds_, retriable_errors_));
  return Status::OK();
}

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewReadOnlyMemoryRegionFromFile(
    const string& filename, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  return RetryingUtils::CallWithRetries(
      [this, &filename, result]() {
        return base_file_system_->NewReadOnlyMemoryRegionFromFile(filename,
                                                                  result);
      },
      initial_delay_microseconds_, retriable_errors_);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_FILE_SYSTEM_H_
