import tensorflow as tf
from tensorflow.python.lib.io import file_io

print(file_io.stat('s3://aws-tensorflow-benchmarking/suthiv-github.tar.gz').length)

from timeit import default_timer as timer

start = timer()
c = len(file_io.get_matching_files('s3://aws-tensorflow-benchmarking/imagenet-armand/*/train*'))
end = timer()
print(end-start, c)