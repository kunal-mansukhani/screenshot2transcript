import tensorflow as tf

build_info = tf.sysconfig.get_build_info()
print("CUDA Version:", build_info['cuda_version'])
print("cuDNN Version:", build_info['cudnn_version'])
print("Number of GPUs:", len(tf.config.list_physical_devices('GPU')))
