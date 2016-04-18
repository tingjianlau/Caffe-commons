#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {

// 所有网络层的基类，定义所有的网络层的通用接口。比如前馈，反馈，Reshape，setup
// 前馈接口，必须实现
// 反馈接口，需要的时候实现，计算梯度
/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   * 每个网络层需要定义它的setup而不需要构造函数
   */
  explicit Layer(const LayerParameter& param)
    : layer_param_(param), is_shared_(false) {
    if(MY_DEBUG)
      LOG(INFO) << "进入Layer类的构造函数";
	// 通过网络层参数来构造网络层
      // Set phase and copy blobs (if there are any).
      phase_ = param.phase(); // 指明是测试还是训练阶段
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          //blobs_的元素是指向Blob<Dtype>的智能指针,需要注意的是这句代码采用的是成员运算符，下一句代码使用的是箭头运算符。reset是因为数据类型Dtype可能会发生变化
          blobs_[i].reset(new Blob<Dtype>());// 清除当前Blob[i]的设置，以便重新写入？？
          blobs_[i]->FromProto(layer_param_.blobs(i)); // 从proto文件中读取数据，其实就是反序列化
        }
      }
    }
  virtual ~Layer() {}

  /**
   * @brief Implements common layer setup functionality. 实现一些通用的设置功能
   *
   * @param bottom the preshaped input blobs 
   *               层的输入数据，blob中的存储空间已申请
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape 层的输出，blob对象已构造但是其中的存储空间未申请，具体在Reshape函数中实现
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types, 调用LayerSetUp来对每一个网络层进行特殊化的处理
   * followed by Reshape to set up sizes of top blobs and internal buffers. 调用Reshape top
   * Sets up the loss weight multiplier blobs for any non-zero loss weights. 设置权重参数
   * This method may not be overridden. 该方法不能被重载
   * 初始化构造函数SetUp
   * 1、检查输入输出blob个数是否满足要求，每个层能处理的输入输出数据不一样
   * 2、调用LayerSetUp函数初始化特殊的层，每个Layer子类需要重写这个函数完成定制的初始化
   * 3、调用Reshape函数为top blob分配合适的存储空间
   * 4、为每个top blob设置loss weight multipler blobs(损失权重乘子blobs), 非LossLayer的top blob的loss weight值为零
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    if(MY_DEBUG)
      LOG(INFO) << "进入Layer类的成员函数SetUp(bottom, top)";
    InitMutex();
    CheckBlobCounts(bottom, top);
    // 假设当前调用LayerSetUp函数的是Data类型的Layer，因为DataLayer没有覆盖其基类的LayerSetUp函数，所以直接调用了从父类继承而来的LayerSetUp函数
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top); //非LossLayer的top blob的loss weight为0
    if(MY_DEBUG)
      LOG(INFO) << "结束Layer类的成员函数SetUp(bottom, top)的调用";
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape. 
   *        定制初始化，每个子类layer必须实现此虚函数！！
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   *     输入blob，数据成员data_和diff_存储了相关数据
   * @param top
   *     the allocated but unshaped output blobs
   *     输出blob，blob对象已构造但数据成员的空间尚未申请
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   * 此方法执行一次定制化的层初始化，包括从layer_param_读入并处
   * 理相关的层权值和偏置值参数，调用Reshape函数申请top blob的存储空间
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Whether a layer should be shared by multiple nets during data
   *        parallelism. By default, all layers except for data layers should
   *        not be shared. data layers should be shared to ensure each worker
   *        solver access data sequentially during data parallelism.
   */
  virtual inline bool ShareInParallel() const { return false; }

  /** @brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  inline bool IsShared() const { return is_shared_; }

  /** @brief Set whether this layer is actually shared by other nets
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then is_shared should be set true.
   */
  inline void SetShared(bool is_shared) {
    CHECK(ShareInParallel() || !is_shared)
        << type() << "Layer does not support sharing.";
    is_shared_ = is_shared;
  }

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs. // 调整top blob和internal buffers以适应bottom blob
   * 定义的层需要实现这个虚函数
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *  给定bottom blobs，计算top blobs 和loss
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   * The Forward wrapper calls the relevant device wrapper function
   *
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   * 
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   * 每个网络需要定义cpu版本的前馈，可选的gpu版本的前馈
   */
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients. 给定top blob的梯度，计算bottom blob的梯度
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index 长度为bottom的个数的向量，每个索引表示是否将损失梯度值反馈到对应索引的bottom中
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
   *  	    返回可学习的参数blobs的向量???
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   * 序列化
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   *        返回指定索引的标量损失值
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   *        设置网络层指定索引位置的loss
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
   *        返回网络层的名字，字符串描述
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *        返回bottom层的blob的确切数目，需要被重写
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *        返回bottom层blob的最小数目？？？
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   * 	    判断bottom和top的数目是否相同
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *        是否自动创建top blob
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   * 	    查询某一个bottom blob是否强制bp
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   * 	    查询一个blob是否bp
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *        设置一个blob是否bp
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }


 protected:
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_; // caffe.proto文件里定义的message 
  /** The phase: TRAIN or TEST */
  Phase phase_;	// 是caffe.ph.h里定义的一个枚举类，指明是训练还是测试阶段
  /** The vector that stores the learnable parameters as a set of blobs. 所定义的向量blobs_里存储的是指向Blob<Dtyp>的智能指针，Blob<Dtype>里面存储的是learnable parameter, 使用向量是因为weight和bias是分开保存再两个blob中。其中blobs_[0]存储当前层的weight,blobs_[1]存储当前层的bias。例如卷积层就会初始化这个参数，数据层就不会*/
  vector<shared_ptr<Blob<Dtype> > > blobs_; 
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_; // 这个bool表示是否为对应的blob计算梯度diff，标志每个top blob是否需要计算反向传递的梯度值

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. 决定每个top blob 在 objective function是否non-zero weigh，即Losslayer中表示每个top blob计算的loss的权重 */
  vector<Dtype> loss_; // 存储top的loss

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;  // cpu版本的前馈实现
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   * 	    gpu版本的前馈实现
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        cpu版本的反馈实现
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   *        gpu版本的反馈实现
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom); // 如果GPU不可以则直接调用cpu版本的反馈实现
  }

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   * 检查bottom和top的大小是否与该层指定的一致
   */
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  } // end of CheckBlobCounts

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   * 用blob初始化损失权重,为每个top blob设置loss weight multipler blobs,非LossLayer的top blob的loss weight为0
   */
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob."; // 如果top.size()和num_loss_weights不相等，则输出错误日志
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }
		this->set_loss(top_id, loss_weight); // 修改Layer的数据成员loss_，其存储的是loss_weight，默认是1
        const int count = top[top_id]->count();
		// 返回指向某块Blob的diff所对应的内存空间指针，并且由于mutable_cpu_diff返回的是void*指针，所以，还有一个类型转换过程
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
		// 用loss_weight初始化loss_multiplier这块内存,其存储count个loss_weight
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  /** Whether this layer is actually shared by other nets*/
  bool is_shared_;

  /** The mutex(互斥锁) for sequential forward if this layer is shared */
  shared_ptr<boost::mutex> forward_mutex_;

  /** Initialize forward_mutex_ */
  void InitMutex();
  /** Lock forward_mutex_ if this layer is shared */
  void Lock();
  /** Unlock forward_mutex_ if this layer is shared */
  void Unlock();

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
// 前馈，根据caffe的mode调用相对应的cpu实现或者是gpu实现，并且计算损失函数的值
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Lock during forward to ensure sequential forward
  Lock(); // 防止死锁???
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
	  // 在LossLayer::Reshape()中设置了其top的shape_为空，但count_为1
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
	  // 这里的loss_weights是SetLossWeights()方法中模板函数caffe_set()初始化的值(default 1)
      const Dtype* loss_weights = top[top_id]->cpu_diff();
	  // 向量內积
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  Unlock(); // 解锁
  return loss;
}

// 反向传播梯度，根据caffe的mode是在gpu还是在cpu，调用相应版本的函数
// propagate_down用于控制对应的层是否bp
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// 序列化网络层参数到协议缓存，最终是调用blob写入协议缓存
// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
