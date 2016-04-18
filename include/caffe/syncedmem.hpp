#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) { // 主要是分配和释放内存
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory { // 定义了内存分配管理和cpu与GPU之间的同步
 public:
// 构造函数及简单的初始化
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
// 重载构造函数，只是设置了size的大小，并未申请内存
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
// 析构函数
  ~SyncedMemory();

// 调用to_cpu()并返回数据在cpu上的指针
  const void* cpu_data();
  void set_cpu_data(void* data);
// 调用to_gpu()并返回数据在gpu上的指针
  const void* gpu_data();
  void set_gpu_data(void* data);
// 调用to_cpu()并改变数据的状态为HEAD_AT_CPU，返回cpu的指针
  void* mutable_cpu_data();
  void* mutable_gpu_data();
// 表示数据的状态的枚举类型，有四种状态，分别是未初始化，数据在cpu中，在GPU中和数据中两者都有
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
// 返回数据的状态
  SyncedHead head() { return head_; }
// 返回数据的大小
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

// 私有成员函数和变量
 private:
// 数据有显存同步到内存
  void to_cpu();
// 数据由内存同步到显存
  void to_gpu();
  void* cpu_ptr_; // 数据在内存的指针
  void* gpu_ptr_; // 数据在显存的指针
  size_t size_;	  // 数据的大小
  SyncedHead head_; // 表示数据的状态，有四种状态，分别是未初始化，数据在cpu中，在GPU中和数据中两者都有
  bool own_cpu_data_; // 是否分配了内存空间，初始值都为false
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_; // gpu的设备号

//一个宏，把该类的拷贝函数和等号操作符号给禁止
  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
