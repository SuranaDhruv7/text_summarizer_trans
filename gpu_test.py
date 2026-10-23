import tensorflow as tf

print("=" * 45)
print("TENSORFLOW AND GPU CHECK")
print("=" * 45)

# 1. Check TensorFlow version
print("\nTensorFlow Version:", tf.__version__)

# 2. Check whether TensorFlow was built with CUDA support
print("Built with CUDA:", tf.test.is_built_with_cuda())

# 3. Detect available GPUs
gpus = tf.config.list_physical_devices("GPU")

print("\nGPU Available:", bool(gpus))
print("Detected GPUs:", gpus)

# 4. Test GPU computation if a GPU is detected
if gpus:
    try:
        with tf.device("/GPU:0"):
            a = tf.random.normal((1000, 1000))
            b = tf.random.normal((1000, 1000))
            c = tf.matmul(a, b)

        print("\nGPU computation test: PASSED")
        print("Tensor result device:", c.device)

    except Exception as e:
        print("\nGPU computation test: FAILED")
        print("Error:", e)

else:
    print("\nTensorFlow cannot currently detect a GPU.")
    print("The computation will need to run on the CPU.")

print("\n" + "=" * 45)