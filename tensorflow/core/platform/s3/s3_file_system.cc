/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <mutex>
#include <thread>

#include "tensorflow/core/platform/s3/s3_file_system.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/s3/aws_crypto.h"
#include "tensorflow/core/platform/s3/aws_logging.h"

#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/utils/FileSystemUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/threading/Executor.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/transfer/TransferManager.h>

#include <cstdlib>

namespace tensorflow {

namespace {
static const char* kS3FileSystemAllocationTag = "S3FileSystemAllocation";
static const size_t kS3ReadAppendableFileBufferSize = 1024 * 1024;
// 5 MB chosen similar to default size of AWS CPP SDK's Transfer Manager
static const size_t kS3MultiPartCopyPartSize = 5 * 1024 * 1024;
static const int kS3GetChildrenMaxKeys = 100;
static const int kExecutorPoolSize = 5;
static const int kUploadRetries = 5;
static const char* kExecutorTag = "TransferManagerExecutor";

Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  static mutex cfg_lock(LINKER_INITIALIZED);
  static bool init(false);
  static Aws::Client::ClientConfiguration cfg;

  std::lock_guard<mutex> lock(cfg_lock);

  if (!init) {
    const char* endpoint = getenv("S3_ENDPOINT");
    if (endpoint) {
      cfg.endpointOverride = Aws::String(endpoint);
    }
    const char* region = getenv("AWS_REGION");
    if (!region) {
      // TODO (yongtang): `S3_REGION` should be deprecated after 2.0.
      region = getenv("S3_REGION");
    }
    if (region) {
      cfg.region = Aws::String(region);
    } else {
      // Load config file (e.g., ~/.aws/config) only if AWS_SDK_LOAD_CONFIG
      // is set with a truthy value.
      const char* load_config_env = getenv("AWS_SDK_LOAD_CONFIG");
      string load_config =
          load_config_env ? str_util::Lowercase(load_config_env) : "";
      if (load_config == "true" || load_config == "1") {
        Aws::String config_file;
        // If AWS_CONFIG_FILE is set then use it, otherwise use ~/.aws/config.
        const char* config_file_env = getenv("AWS_CONFIG_FILE");
        if (config_file_env) {
          config_file = config_file_env;
        } else {
          const char* home_env = getenv("HOME");
          if (home_env) {
            config_file = home_env;
            config_file += "/.aws/config";
          }
        }
        Aws::Config::AWSConfigFileProfileConfigLoader loader(config_file);
        loader.Load();
        auto profiles = loader.GetProfiles();
        if (!profiles["default"].GetRegion().empty()) {
          cfg.region = profiles["default"].GetRegion();
        }
      }
    }
    const char* use_https = getenv("S3_USE_HTTPS");
    if (use_https) {
      if (use_https[0] == '0') {
        cfg.scheme = Aws::Http::Scheme::HTTP;
      } else {
        cfg.scheme = Aws::Http::Scheme::HTTPS;
      }
    }
    const char* verify_ssl = getenv("S3_VERIFY_SSL");
    if (verify_ssl) {
      if (verify_ssl[0] == '0') {
        cfg.verifySSL = false;
      } else {
        cfg.verifySSL = true;
      }
    }
    const char* connect_timeout = getenv("S3_CONNECT_TIMEOUT_MSEC");
    if (connect_timeout) {
      int64 timeout;

      if (strings::safe_strto64(connect_timeout, &timeout)) {
        cfg.connectTimeoutMs = timeout;
      }
    }
    const char* request_timeout = getenv("S3_REQUEST_TIMEOUT_MSEC");
    if (request_timeout) {
      int64 timeout;

      if (strings::safe_strto64(request_timeout, &timeout)) {
        cfg.requestTimeoutMs = timeout;
      }
    }
    const char* ca_file = getenv("S3_CA_FILE");
    if (ca_file) {
      cfg.caFile = Aws::String(ca_file);
    }
    const char* ca_path = getenv("S3_CA_PATH");
    if (ca_path) {
      cfg.caPath = Aws::String(ca_path);
    }
    init = true;
  }

  return cfg;
};

void ShutdownClient(Aws::S3::S3Client* s3_client) {
  if (s3_client != nullptr) {
    delete s3_client;
    Aws::SDKOptions options;
    Aws::ShutdownAPI(options);
    AWSLogSystem::ShutdownAWSLogging();
  }
}

void ShutdownTransferManager(Aws::Transfer::TransferManager* transfer_manager) {
  if (transfer_manager != nullptr) {
    delete transfer_manager;
  }
}

void ShutdownExecutor(Aws::Utils::Threading::PooledThreadExecutor* executor) {
  if (executor != nullptr) {
    delete executor;
  }
}

Status ParseS3Path(const string& fname, bool empty_object_ok, string* bucket,
                   string* object) {
  if (!bucket || !object) {
    return errors::Internal("bucket and object cannot be null.");
  }
  StringPiece scheme, bucketp, objectp;
  io::ParseURI(fname, &scheme, &bucketp, &objectp);
  if (scheme != "s3") {
    return errors::InvalidArgument("S3 path doesn't start with 's3://': ",
                                   fname);
  }
  *bucket = string(bucketp);
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument("S3 path doesn't contain a bucket name: ",
                                   fname);
  }
  str_util::ConsumePrefix(&objectp, "/");
  *object = string(objectp);
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument("S3 path doesn't contain an object name: ",
                                   fname);
  }
  return Status::OK();
}

class S3RandomAccessFile : public RandomAccessFile {
 public:
  S3RandomAccessFile(const string& bucket, const string& object,
                     std::shared_ptr<Aws::S3::S3Client> s3_client)
      : bucket_(bucket), object_(object), s3_client_(s3_client) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    VLOG(1) << "ReadFilefromS3 s3://" << bucket_ << "/" << object_ << " from " << offset << " for n:" << n;
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.WithBucket(bucket_.c_str()).WithKey(object_.c_str());
    string bytes = strings::StrCat("bytes=", offset, "-", offset + n - 1);
    getObjectRequest.SetRange(bytes.c_str());
    getObjectRequest.SetResponseStreamFactory([]() {
      return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag);
    });
    auto getObjectOutcome = this->s3_client_->GetObject(getObjectRequest);
    if (!getObjectOutcome.IsSuccess()) {
      auto error = getObjectOutcome.GetError();
      if (error.GetResponseCode() == Aws::Http::HttpResponseCode::REQUESTED_RANGE_NOT_SATISFIABLE) {
        n = 0;
        *result = StringPiece(scratch, n);
        return Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else {
        return errors::Unknown(error.GetExceptionName(), error.GetMessage());
      }
    }
    n = getObjectOutcome.GetResult().GetContentLength();
    getObjectOutcome.GetResult().GetBody().read(scratch, n);
    *result = StringPiece(scratch, n);
    return Status::OK();
  }

 private:
  string bucket_;
  string object_;
  std::shared_ptr<Aws::S3::S3Client> s3_client_;
};

static int tmp_file_index_;
static std::mutex tmp_file_suffix_lock_;

class S3WritableFile : public WritableFile {
 public:
  S3WritableFile(
      const string& bucket, const string& object,
      std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager,
      std::shared_ptr<Aws::S3::S3Client> s3_client)
      : bucket_(bucket),
        object_(object),
        transfer_manager_(transfer_manager),
        s3_client_(s3_client),
        sync_needed_(true),
        outfile_(Aws::MakeShared<Aws::Utils::TempFile>(
            kS3FileSystemAllocationTag, GetTmpFileSuffix().c_str(),
            std::ios_base::binary | std::ios_base::trunc | std::ios_base::in |
                std::ios_base::out)) {}

  Status Append(StringPiece data) override {
    if (!outfile_) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    sync_needed_ = true;
    outfile_->write(data.data(), data.size());
    if (!outfile_->good()) {
      return errors::Internal(
          "Could not append to the internal temporary file.");
    }
    return Status::OK();
  }

  Status Close() override {
    if (outfile_) {
      TF_RETURN_IF_ERROR(Sync());
      outfile_.reset();
    }
    return Status::OK();
  }

  Status Flush() override { return Sync(); }

  Status Sync() override {
    if (!outfile_) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    if (!sync_needed_) {
      return Status::OK();
    }
    VLOG(1) << "WriteFileToS3: s3://" << bucket_ << "/" << object_;
    long offset = outfile_->tellp();
    std::shared_ptr<Aws::Transfer::TransferHandle> handle =
        transfer_manager_.get()->UploadFile(
            outfile_, bucket_.c_str(), object_.c_str(),
            "application/octet-stream", Aws::Map<Aws::String, Aws::String>());
    handle->WaitUntilFinished();
    int retries = 0;
    while (handle->GetStatus() == Aws::Transfer::TransferStatus::FAILED &&
           retries++ < kUploadRetries) {
      // if multipart upload was used, only the failed parts will be re-sent
      VLOG(1) << "Retrying Upload of s3://" << bucket_ << "/" << object_
              << " after failure. Current retry count:" << retries;
      transfer_manager_.get()->RetryUpload(outfile_, handle);
      handle->WaitUntilFinished();
    }
    if (handle->GetStatus() != Aws::Transfer::TransferStatus::COMPLETED) {
      return errors::Unknown(handle->GetLastError().GetExceptionName(), ": ",
                             handle->GetFailedParts().size(), " failed parts. ",
                             handle->GetLastError().GetMessage());
    }
    outfile_->clear();
    outfile_->seekp(offset);
    return Status::OK();
  }

 private:
  string bucket_;
  string object_;
  bool sync_needed_;
  std::shared_ptr<Aws::Utils::TempFile> outfile_;
  std::shared_ptr<Aws::S3::S3Client> s3_client_;
  std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;

  std::string GetTmpFileSuffix() {
    const int max_tmp_file_index = 1000;
    int tmp_file_index;
    {
      std::lock_guard<std::mutex> lock(tmp_file_suffix_lock_);
      tmp_file_index = tmp_file_index_;
      tmp_file_index_ = (tmp_file_index_ + 1) % max_tmp_file_index;
    }
    std::ostringstream tmp_file_suffix;
    tmp_file_suffix << "/tmp/s3_filesystem_XXXXXX" << tmp_file_index << "_";
    return tmp_file_suffix.str();
  }
};

class S3ReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  S3ReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

}  // namespace

S3FileSystem::S3FileSystem()
    : s3_client_(nullptr, ShutdownClient),
      initialization_lock_(),
      transfer_manager_(nullptr, ShutdownTransferManager),
      executor_(nullptr, ShutdownExecutor) {}

S3FileSystem::~S3FileSystem() {}

// Initializes s3_client_, if needed, and returns it.
std::shared_ptr<Aws::S3::S3Client> S3FileSystem::GetS3Client() {
  std::lock_guard<mutex> lock(this->initialization_lock_);

  if (this->s3_client_.get() == nullptr) {
    AWSLogSystem::InitializeAWSLogging();

    Aws::SDKOptions options;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256Factory>(AWSCryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256HmacFactory>(AWSCryptoAllocationTag);
    };
    options.cryptoOptions.secureRandomFactory_create_fn = []() {
      return Aws::MakeShared<AWSSecureRandomFactory>(AWSCryptoAllocationTag);
    };
    Aws::InitAPI(options);

    // The creation of S3Client disables virtual addressing:
    //   S3Client(clientConfiguration, signPayloads, useVirtualAdressing = true)
    // The purpose is to address the issue encountered when there is an `.`
    // in the bucket name. Due to TLS hostname validation or DNS rules,
    // the bucket may not be resolved. Disabling of virtual addressing
    // should address the issue. See GitHub issue 16397 for details.
    this->s3_client_ = std::shared_ptr<Aws::S3::S3Client>(new Aws::S3::S3Client(
        GetDefaultClientConfig(),
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false));
  }

  return this->s3_client_;
}

std::shared_ptr<Aws::Transfer::TransferManager>
S3FileSystem::GetTransferManager() {
  std::shared_ptr<Aws::S3::S3Client> s3_client = this->GetS3Client();
  std::lock_guard<mutex> lock(this->initialization_lock_);
  if (this->transfer_manager_.get() == nullptr) {
    Aws::Transfer::TransferManagerConfiguration config(
        this->GetExecutor().get());
    config.s3Client = s3_client;
    this->transfer_manager_ = Aws::Transfer::TransferManager::Create(config);
  }
  return this->transfer_manager_;
}

std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
S3FileSystem::GetExecutor() {
  if (this->executor_.get() == nullptr) {
    this->executor_ =
        Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
            kExecutorTag, kExecutorPoolSize);
  }
  return this->executor_;
}

Status S3FileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3RandomAccessFile(bucket, object, this->GetS3Client()));
  return Status::OK();
}

Status S3FileSystem::NewWritableFile(const string& fname,
                                     std::unique_ptr<WritableFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3WritableFile(bucket, object, this->GetTransferManager(),
                                   this->GetS3Client()));
  return Status::OK();
}

Status S3FileSystem::NewAppendableFile(const string& fname,
                                       std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<RandomAccessFile> reader;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &reader));
  std::unique_ptr<char[]> buffer(new char[kS3ReadAppendableFileBufferSize]);
  Status status;
  uint64 offset = 0;
  StringPiece read_chunk;

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3WritableFile(bucket, object, this->GetTransferManager(),
                                   this->GetS3Client()));

  while (true) {
    status = reader->Read(offset, kS3ReadAppendableFileBufferSize, &read_chunk,
                          buffer.get());
    if (status.ok()) {
      (*result)->Append(read_chunk);
      offset += kS3ReadAppendableFileBufferSize;
    } else if (status.code() == error::OUT_OF_RANGE) {
      (*result)->Append(read_chunk);
      break;
    } else {
      (*result).reset();
      return status;
    }
  }

  return Status::OK();
}

Status S3FileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(fname, &size));
  std::unique_ptr<char[]> data(new char[size]);

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &file));

  StringPiece piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

  result->reset(new S3ReadOnlyMemoryRegion(std::move(data), size));
  return Status::OK();
}

Status S3FileSystem::FileExists(const string& fname) {
  FileStatistics stats;
  TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
  return Status::OK();
}

Status S3FileSystem::GetChildren(const string& dir,
                                 std::vector<string>* result) {
  VLOG(1) << "GetChildren for path: " << dir;
  string bucket, prefix;
  TF_RETURN_IF_ERROR(ParseS3Path(dir, false, &bucket, &prefix));

  if (prefix.back() != '/') {
    prefix.push_back('/');
  }

  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(kS3GetChildrenMaxKeys)
      .WithDelimiter("/");
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult listObjectsResult;
  do {
    auto listObjectsOutcome =
        this->GetS3Client()->ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      return errors::Unknown(listObjectsOutcome.GetError().GetExceptionName(),
                             ": ", listObjectsOutcome.GetError().GetMessage());
    }

    listObjectsResult = listObjectsOutcome.GetResult();
    for (const auto& object : listObjectsResult.GetCommonPrefixes()) {
      Aws::String s = object.GetPrefix();
      s.erase(s.length() - 1);
      Aws::String entry = s.substr(strlen(prefix.c_str()));
      if (entry.length() > 0) {
        result->push_back(entry.c_str());
      }
    }
    for (const auto& object : listObjectsResult.GetContents()) {
      Aws::String s = object.GetKey();
      Aws::String entry = s.substr(strlen(prefix.c_str()));
      if (entry.length() > 0) {
        result->push_back(entry.c_str());
      }
    }
    listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
  } while (listObjectsResult.GetIsTruncated());

  return Status::OK();
}

Status S3FileSystem::Stat(const string& fname, FileStatistics* stats) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, true, &bucket, &object));
  VLOG(1) << "Stat on path: " << fname;
  if (object.empty()) {
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket.c_str());
    auto headBucketOutcome = this->GetS3Client()->HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      return errors::Unknown(headBucketOutcome.GetError().GetExceptionName(),
                             ": ", headBucketOutcome.GetError().GetMessage());
    }
    stats->length = 0;
    stats->is_directory = 1;
    return Status::OK();
  }

  bool found = false;

  Aws::S3::Model::HeadObjectRequest headObjectRequest;
  headObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());
  headObjectRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto headObjectOutcome = this->GetS3Client()->HeadObject(headObjectRequest);
  if (headObjectOutcome.IsSuccess()) {
    stats->length = headObjectOutcome.GetResult().GetContentLength();
    stats->is_directory = 0;
    stats->mtime_nsec =
        headObjectOutcome.GetResult().GetLastModified().Millis() * 1e6;
    found = true;
  }
  string prefix = object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }
  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(1);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto listObjectsOutcome =
      this->GetS3Client()->ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    auto listObjects = listObjectsOutcome.GetResult().GetContents();
    if (listObjects.size() > 0) {
      stats->length = 0;
      stats->is_directory = 1;
      stats->mtime_nsec = listObjects[0].GetLastModified().Millis() * 1e6;
      found = true;
    }
  }
  if (!found) {
    return errors::NotFound("Object ", fname, " does not exist");
  }
  return Status::OK();
}

Status S3FileSystem::GetMatchingPaths(const string& pattern,
                                      std::vector<string>* results) {
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status S3FileSystem::DeleteFile(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  VLOG(1) << "DeleteFile: " << fname;
  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());

  auto deleteObjectOutcome =
      this->GetS3Client()->DeleteObject(deleteObjectRequest);
  if (!deleteObjectOutcome.IsSuccess()) {
    return errors::Unknown(deleteObjectOutcome.GetError().GetExceptionName(),
                           ": ", deleteObjectOutcome.GetError().GetMessage());
  }
  return Status::OK();
}

Status S3FileSystem::CreateDir(const string& dirname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, true, &bucket, &object));
  VLOG(1) << "CreateDir: " << dirname;
  if (object.empty()) {
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket.c_str());
    auto headBucketOutcome = this->GetS3Client()->HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      return errors::NotFound("The bucket ", bucket, " was not found.");
    }
    return Status::OK();
  }
  string filename = dirname;
  if (filename.back() != '/') {
    filename.push_back('/');
  }
  if (!this->FileExists(filename).ok()) {
    std::unique_ptr<WritableFile> file;
    TF_RETURN_IF_ERROR(NewWritableFile(filename, &file));
    TF_RETURN_IF_ERROR(file->Close());
  }
  return Status::OK();
}

Status S3FileSystem::DeleteDir(const string& dirname) {
  VLOG(1) << "DeleteDir: " << dirname;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, false, &bucket, &object));

  string prefix = object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }
  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(2);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto listObjectsOutcome =
      this->GetS3Client()->ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    auto contents = listObjectsOutcome.GetResult().GetContents();
    if (contents.size() > 1 ||
        (contents.size() == 1 && contents[0].GetKey() != prefix.c_str())) {
      // Due to Eventual consistency of S3, list may return the objects even after the deletes above
      // to retry this operation in such case, the error type has been changed to Errors::Internal
      return errors::Internal("Cannot delete a non-empty directory.");
    }
    if (contents.size() == 1 && contents[0].GetKey() == prefix.c_str()) {
      string filename = dirname;
      if (filename.back() != '/') {
        filename.push_back('/');
      }
      return DeleteFile(filename);
    }
  }
  return Status::OK();
}

Status S3FileSystem::GetFileSize(const string& fname, uint64* file_size) {
  FileStatistics stats;
  TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
  *file_size = stats.length;
  return Status::OK();
}

Status S3FileSystem::MultiPartCopy(Aws::String source_path, const Aws::String& target_bucket, const Aws::String& target_key) {
  VLOG(1) << "MultiPartCopy from " << source_path << " to: s3://" << target_bucket <<"/" << target_key;

  Aws::S3::Model::CreateMultipartUploadRequest multipartUploadRequest;
  multipartUploadRequest.SetBucket(target_bucket);
  multipartUploadRequest.SetKey(target_key);

  auto multipartUploadOutcome = this->GetS3Client()->CreateMultipartUpload(multipartUploadRequest);
  if (!multipartUploadOutcome.IsSuccess())
  {
    return errors::Unknown(multipartUploadOutcome.GetError().GetExceptionName(), ": ", multipartUploadOutcome.GetError().GetMessage())
  }

  Aws::String uploadID = multipartUploadOutcome.GetResult().GetUploadId();

  FileStatistics stats;
  TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
  int numParts = stats.length / kS3MultiPartCopyPartSize

  Aws::S3::Model::CompletedMultipartUpload completedMPURequest;

  for (int partNumber=0; partNumber<num_parts; partNumber++) {
    int retryCount = 3;
    while (retryCount > 0) {
      retryCount--;
      
      uint64 startPos = (partNumber - 1) * kS3MultiPartCopyPartSize;
      uint64 endPos = startPos + kS3MultiPartCopyPartSize - 1;

      std::ostringstream rangeStream;
      rangeStream << "bytes=" << startPos << std::to_string(startPos) << "-" << std::to_string(endPos);
      string range = rangeStream.str();

      Aws::S3::Model::UploadPartCopyRequest uploadPartCopyRequest;
      uploadPartCopyRequest.SetBucket(target_bucket);
      uploadPartCopyRequest.SetKey(target_key);

      source_path = Aws::Utils::StringUtils::URLEncode(source_path.c_str());

      uploadPartCopyRequest.SetCopySource(source_path.c_str());
      uploadPartCopyRequest.SetCopySourceRange(range.c_str());
      uploadPartCopyRequest.SetPartNumber(partNumber);
      uploadPartCopyRequest.SetUploadId(uploadID);

      auto uploadPartCopyOutcome = this->GetS3Client()->UploadPartCopy(uploadPartCopyRequest);
      if (!uploadPartCopyOutcome.IsSuccess() && retryCount > 0) {
        LOG(INFO) << "Retrying failed copy of part " << std::to_string(partNumber) << " during multi part copy from "
                  << source_path << " to s3://" << target_bucket << "/" << target_key << " failed.";
      } else if (!uploadPartCopyOutcome.IsSuccess()) {
        return errors::Unknown(uploadPartCopyOutcome.GetError().GetExceptionName(), ": ",
                               uploadPartCopyOutcome.GetError().GetMessage());
      }

      Aws::String sETag = uploadPartCopyOutcome.GetResult().GetCopyPartResult().GetETag();
      Aws::S3::Model::CompletedPart completedPart;
      completedPart.SetPartNumber(iPartNumber);
      completedPart.SetETag(sETag);
      completedMPURequest.AddParts(completedPart);
    }
    
    Aws::S3::Model::CompleteMultipartUploadRequest completeRequest;
    completeRequest.SetBucket(target_bucket);
    completeRequest.SetKey(target_key);
    completeRequest.SetUploadId(uploadID);
    completeRequest.SetMultipartUpload(completedMPURequest);
    auto completeOutcome = pS3Client->CompleteMultipartUpload(completeRequest);
    if (!completeOutcome.IsSuccess()) {
      return errors::Unknown(completeOutcome.GetError().GetExceptionName(), ": ",
                             completeOutcome.GetError().GetMessage());
    }
  }
  
}

Status S3FileSystem::RenameFile(const string& src, const string& target) {
  VLOG(1) << "RenameFile from: " << src << " to: " << target;
  string src_bucket, src_object, target_bucket, target_object;
  TF_RETURN_IF_ERROR(ParseS3Path(src, false, &src_bucket, &src_object));
  TF_RETURN_IF_ERROR(
      ParseS3Path(target, false, &target_bucket, &target_object));
  if (src_object.back() == '/') {
    if (target_object.back() != '/') {
      target_object.push_back('/');
    }
  } else {
    if (target_object.back() == '/') {
      target_object.pop_back();
    }
  }

  Aws::S3::Model::CopyObjectRequest copyObjectRequest;
  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;

  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(src_bucket.c_str())
      .WithPrefix(src_object.c_str())
      .WithMaxKeys(kS3GetChildrenMaxKeys);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult listObjectsResult;
  do {
    auto listObjectsOutcome =
        this->GetS3Client()->ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      return errors::Unknown(listObjectsOutcome.GetError().GetExceptionName(),
                             ": ", listObjectsOutcome.GetError().GetMessage());
    }

    listObjectsResult = listObjectsOutcome.GetResult();
    for (const auto& object : listObjectsResult.GetContents()) {
      Aws::String src_key = object.GetKey();
      Aws::String target_key = src_key;
      target_key.replace(0, src_object.length(), target_object.c_str());
      Aws::String source = Aws::String(src_bucket.c_str()) + "/" +
                           Aws::Utils::StringUtils::URLEncode(src_key.c_str());

      auto copyObjectOutcome = MultiPartCopy(source, Aws::String(target_bucket.c_str()), target_key);
      if (!copyObjectOutcome.IsSuccess()) {
        return errors::Unknown(copyObjectOutcome.GetError().GetExceptionName(),
                               ": ", copyObjectOutcome.GetError().GetMessage());
      }
      
      deleteObjectRequest.SetBucket(src_bucket.c_str());
      deleteObjectRequest.SetKey(src_key.c_str());

      auto deleteObjectOutcome =
          this->GetS3Client()->DeleteObject(deleteObjectRequest);
      if (!deleteObjectOutcome.IsSuccess()) {
        return errors::Unknown(
            deleteObjectOutcome.GetError().GetExceptionName(), ": ",
            deleteObjectOutcome.GetError().GetMessage());
      }
    }
    listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
  } while (listObjectsResult.GetIsTruncated());

  return Status::OK();
}

Status S3FileSystem::NeedsTempLocation(const string& path) {
  return Status(tensorflow::error::FAILED_PRECONDITION,
                "Does not need a temp location");
}

REGISTER_FILE_SYSTEM("s3", RetryingS3FileSystem);

}  // namespace tensorflow
