#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param, const Net* root_net = NULL);
  explicit Net(const string& param_file, Phase phase,
      const Net* root_net = NULL);
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward with the input Blob%s already fed separately.
   *
   * You can get the input blobs using input_blobs().
   */
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL);

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief Run forward using a set of bottom blobs, and return the result.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);
  /**
   * @brief Run forward using a serialized BlobProtoVector and return the
   *        result as a serialized BlobProtoVector
   */
  string Forward(const string& input_blob_protos, Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward(const vector<Blob<Dtype>* > & bottom) {
    Dtype loss;
    Forward(bottom, &loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int>& param_owners() const { return param_owners_; }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

 protected:
  // Helpers for Init.
  /// @brief Append a new input or top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward about input Blobs.
  void InputDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net net里的各层，这里会记录实体，名字，序号和是否后向
  vector<shared_ptr<Layer<Dtype> > > layers_; 
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;

  /// @brief the blobs storing intermediate results between the layer. net 中所有blob
  /// 这里所说的所有非参数blob其实指的是AppendTop函数中遍历的top blob，并不是每一层top+bottom，因为这一层的top就是下一层的bottom
  vector<shared_ptr<Blob<Dtype> > > blobs_; // 存储的中间的结果，是针对整个网络中的所有非参数blob而设计的一个变量
  vector<string> blob_names_; // 整个网络中，所有的非参数blob的那么
  map<string, int> blob_names_index_; // 制定顺序是：1,txt中的layer,2每个layer中的bottom和top
  vector<bool> blob_need_backward_; // 所有的blob是否需要backward

  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers. 上面记录了所有的blob，即外面一层的vector装载的是哪个layer的Blob，而里面的vector存储的bottom的用途，比如SoftmaxWithLoss层中的bottom[0]存储上一层的top[0]，而bottom[1]存储的是从Data层传过来的label
  vector<vector<Blob<Dtype>*> > bottom_vecs_; //存储整个网络所有网络层的bottom blob指针，实际上存储的是前一层的top指针
  vector<vector<int> > bottom_id_vecs_; // 存储整个网络所有网络层的bottom blob的ID
  vector<vector<bool> > bottom_need_backward_; // 整个网络所有网络层的bottom blob是否需要backward

  /// top_vecs stores the vectors containing the output for each layer
  /// 存储整个网络所有网络层的top blob的指针
  vector<vector<Blob<Dtype>*> > top_vecs_;
  // 存储整个网络所有网络层的top blob的ID。top_id_vecs_中存储的最基本元素是blob_id，每一个新的blob都会赋予一个blob_id，top_vecs则与之对应。但这个id可能会有重复（因为in-place)
  vector<vector<int> > top_id_vecs_;

  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_; // 每次遍历一个layer的时候，都会resize blob_loss_weight_，然后调用模板类layer的loss函数返回loss_weight
  vector<vector<int> > param_id_vecs_; //存储的基本元素是net_param_id，每遍历一个参数blob,net_param_id和param_id_vecs_都会更新
  vector<int> param_owners_; // 是一个存储parameter "onwer"的一个向量  ——> -1 表示当前Layer就是该parameter的"owner"
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;//其元素为当layer_id 与当前param_id 组成的pair
  map<string, int> param_names_index_; //是整个网络的参数non-empty name与index的映射。注意，这个name是ParamSpec 类型中的name。

  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_; // 整个网络的输入，在Forward更新
  vector<Blob<Dtype>*> net_output_blobs_; // 整个网络的输出，例如训练层只输出loss，测试层输出loss和accuracy blob
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  /// The root net that actually holds the shared layers in data parallelism
  const Net* const root_net_; // 默认是Null
  DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
