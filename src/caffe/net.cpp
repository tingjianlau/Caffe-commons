#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net) {
  if(MY_DEBUG)
    LOG(INFO) << "调用Net构造函数成功";
  // 调用Init初始化网络
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const Net* root_net)
    : root_net_(root_net) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  param.mutable_state()->set_phase(phase);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  if(MY_DEBUG)
    LOG(INFO) << "调用Init初始化网络";
  CHECK(Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase(); // 给数据成员赋值
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param; // 存储符合要求的层的参数
  // 比较当前的phase/level/stage，移除不符合要求的layer
  FilterNet(in_param, &filtered_param);
  if(MY_DEBUG)
	LOG(INFO) << "FilterNet前后网络的剩余的层数对比: " << in_param.layer_size() << " " << filtered_param.layer_size();
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString(); // 打印经过过滤后的所有层的参数信息
  if(MY_DEBUG)
    LOG(INFO) << "经过过滤的所有层的信息打印结束";
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  // 插入分离的层，如测试网络时插入了label_mnist_1_split层
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name(); // 给数据成员赋值
  //blob_name_to_idx是一个局部变量，其实它是在当前layer的top blob 和下一层的bottom blob间起着一个桥梁作用。
  //blob_name_to_idx中元素的pair是从网络最开始一层一层搭建的过程中压入map的，其中的name和id都是不重复的。name是关键字——不重复是map数据结构的必然要求，id也是不重复的——0,1,2...
  map<string, int> blob_name_to_idx;
  set<string> available_blobs; // 主要用来更新数据成员net_output_blobs_
  if(MY_DEBUG)
    LOG(INFO) << "param.input_dim_size(): " << param.input_dim_size() << " param.input_shape_size(): " << param.input_shape_size()<< " param.input_size(): " << param.input_size();

  CHECK(param.input_dim_size() == 0 || param.input_shape_size() == 0)
      << "Must specify either input_shape OR deprecated input_dim, not both.";
  if (param.input_dim_size() > 0) {
    // Deprecated 4D dimensions.
    CHECK_EQ(param.input_size() * 4, param.input_dim_size())
        << "Incorrect input blob dimension specifications.";
  } else {
    CHECK_EQ(param.input_size(), param.input_shape_size())
        << "Exactly one input_shape must be specified per input.";
  }
  memory_used_ = 0; // 初始化net使用的内存空间大小
  // set the input blobs
  for (int input_id = 0; input_id < param.input_size(); ++input_id) {
    if(MY_DEBUG)
      LOG(INFO) << "测试是否进入for循环,idx = " << input_id;
    const int layer_id = -1;  // inputs have fake layer ID -1
    AppendTop(param, layer_id, input_id, &available_blobs, &blob_name_to_idx);
  }
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size()); // 用当前网络的层数设置为bottom_vecs_的大小
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  if(MY_DEBUG)
	LOG(INFO) << "当前网络的层数：" << param.layer_size();
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) { //对于没有phase阈的层，设置当前phase
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer. 声明一个LayerParameter类的对象，用来装载当前层的参数信息
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      // 当存在某些层的bottom应该被跳过时，检查这个值是否符合要求
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    if (share_from_root) {
      LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
      layers_.push_back(root_net_->layers_[layer_id]);
      layers_[layer_id]->SetShared(true);
    } else {
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    }
    layer_names_.push_back(layer_param.name()); //将当前层的name加入容器layer_names_中
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    if(MY_DEBUG)
      LOG(INFO) << "this layer has " << layer_param.bottom_size() << " bottom blob and " << layer_param.top_size() << " top blob";
    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      // 在遍历所有的bottom_id的过程中，只要有一次使得need_backward为真，则这个for循环结束后，need_backward也为真
      need_backward |= blob_need_backward_[blob_id];
      if(MY_DEBUG)
        LOG(INFO) << "第" << layer_id << "层的blob_need_backward_[" << blob_id << "]: " << blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
	  if(MY_DEBUG)
		LOG(INFO) << "needed_num_top: " << needed_num_top << " (AutoToplobs)";
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      // 利用工厂函数根据LayerParameter可以得到一个指向基类Layer的智能指针shared_ptr<Layer<Dtype>>,这里使用指针是为了利用C++的多态性————基类指针既可以与基类绑定，也可以绑定派生类对象，如果在这里真正用到的是DataLayer，那么该指针应该与DataLayer对象绑定。这关系到SetUp函数中根据Layer的类型调用不同的类型的LayerSetUp派生类!!!
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    // 每次循环都会更新向量blob_loss_weights_
    if(MY_DEBUG)
      LOG(INFO) << "top_vecs_[layer_id].size(): " << top_vecs_[layer_id].size() << " blob_loss_weights_.size(): " << blob_loss_weights_.size();
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        // 默认用0填充
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      // loss函数返回loss_weight ——> 在模板类的SetUp方法中会调用SetLossWeights来设置其私有数据成员loss_,里面存储的其实是loss_weight
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      // 打印Top blob的信息，例如包括data和label
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      // 累加所需内存的大小
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    // 打印每个top blob所需的内存大小
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    if(MY_DEBUG)
      LOG(INFO) << "layer_param.param_size(): " << layer_param.param_size() << "  layers_[" << layer_id << "]->blobs().size()--可学习的参数: " << layers_[layer_id]->blobs().size();
    const int param_size = layer_param.param_size();
    // 返回可学习的参数
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    //param_size是Layermeter类型对象layer_param中ParamSpec param成员的个数, num_param_blobs是一个Layer中learnable parameter blob的个数，param_size <= num_param_blobs ???
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    // 声明一个ParamSpec类型的对象
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      // 获取对应ParamSpec的对象
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      // 当lr_mult不等于0时，就表示这个参数需要反馈调节
      const bool param_need_backward = param_spec->lr_mult() != 0;
      // 只要有一次 param_need_backward为真，则need_backward就为真
      need_backward |= param_need_backward;
      if(MY_DEBUG)
        LOG(INFO) << "对比lr_mult信息后后判断当前层是否需要backward: " << need_backward;
      // 调用Layer类的成员函数set_param_propagate_down设置当前层的parameter blob是否需要计算diff backward
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      // 这里的param是局部变量指的是经过过滤后的所有层的参数信息
      // 添加parameter blob，如果当前layer没有parameter blob(num_param_blobs==0),比如RELU，那么就不进入循环，不添加parameter blob
      // AppendParam只是执行为当前layer添加parameter blob的相关工作，并不会修改与backward的相关属性
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward); // 初始化数据成员layer_need_backward_
    //在上述的AppendTop函数中，在遍历当前层的每一个top blob的时候都会将一个false（默认值）压入向量blob_need_backward_。在下面的代码中，如果这个layer need backward，则会更新blob_need_backward_
    if (need_backward) { 
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  } // end of for
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  // 需要注意的是，上述代码中关于backward设置的部分，是按照前向的顺序设置的，而下面的代码是按后向顺序修正前向设置的结果。
  // 一个layer是否需要backward computation，主要依据两个方面：(1)该layer的top blob 是否参与loss的计算；(2):该layer的bottom blob 是否需要backward computation，比如Data层一般就不需要backward computation
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
} // end of Init

// 给定当前的phase/level/stage，移除不符合要求的layer，比如：当当// 前是TRAIN阶段是，就必须移除mnist和accuracy层，因为这两层都是TEST阶段才使用的
template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param); // 拷贝message
  param_filtered->clear_layer(); // clear每个层的fields并设为默认值，不释放内存
  if(MY_DEBUG)
    LOG(INFO) << "param.layer_size():" << param.layer_size();
  // 循环检查每个层，判断是否应该移除
  for (int i = 0; i < param.layer_size(); ++i) {
    // 取第i层
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    
    if(MY_DEBUG)
      LOG(INFO) << "layer_param.include_size()：" << layer_param.include_size() <<  " execlude_size(): " << layer_param.exclude_size();
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      // StateMeetRule()中的net的state是否满足NetStateRule
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        // 如果不包含include，只要meet一个include_size(idx)即可
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        // 如果包含include,只要符合一个include_size(idx)即可
        layer_included = true;
      }
    }
    // 如果当前层满足所有要求，则加入该层
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  // 检查当前层的phase是否与已经设置好的phase相同
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new input or top blob to the net.  (Inputs have
// layer_id == -1, tops have layer_id >= 0.)
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  if(MY_DEBUG)
    LOG(INFO) << "进入AppendTop()函数";
 // 如果layer_id大于0则新建一个LayerParameter对象
  shared_ptr<LayerParameter> layer_param((layer_id >= 0) ?
    (new LayerParameter(param.layer(layer_id))) : NULL);
  // 获取当前要加入layer的blob的name
  const string& blob_name = layer_param ?
      (layer_param->top_size() > top_id ?
          layer_param->top(top_id) : "(automatic)") : param.input(top_id);
  
  if(MY_DEBUG)
    LOG(INFO) << "blob_name: " << blob_name;
  // Check if we are doing in-place computation ????
  if (blob_name_to_idx && layer_param && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
	// 条件blob_name == layer_param->bottom(top_id)这个是关键所在，例如在ReLU层这个条件就满足，因为其top和bottom都是"ip1"
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
	// 将上一层即ip1层的top直接当做这一层的top压入
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      if (layer_param) {
        // 打印当前层的name和当前要加入的blob_name
        LOG(INFO) << layer_param->name() << " -> " << blob_name << " (Normal ouput) ";
      } else {
        LOG(INFO) << "Input " << top_id << " -> " << blob_name << " (Normal ouput) ";
      }
    }
    // 调用Blob的构造函数，新建一个shared_ptr类型的Blob指针
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    if(MY_DEBUG)
      LOG(INFO) << "blobs_.size(),即局部变量blob_id: " << blobs_.size();
    const int blob_id = blobs_.size(); // blob_id是从0到top_size()逐渐增加的，因为每次循环增加top_id时，都会push_back一个新的Blob对象，所以blobs_.size()也会增加1
    blobs_.push_back(blob_pointer); //将新建的Blob对象指针加入数据成员blobs_中
    blob_names_.push_back(blob_name); // 更新数据成员blob_names_
    blob_need_backward_.push_back(false); // 跟新数据成员blob_need_backward
    //blob_name_to_idx和blobs_一样，在"Normal output"的情形下，每次遍历到一个top blob的时候都会更新
    if(MY_DEBUG)
      LOG(INFO) << "(*blob_name_to_idx)[blob_name]: " << (*blob_name_to_idx)[blob_name]; 
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    if (layer_id == -1) {
      // Set the (explicitly specified) dimensions of the input blob.
      if (param.input_dim_size() > 0) {
        blob_pointer->Reshape(param.input_dim(top_id * 4),
                              param.input_dim(top_id * 4 + 1),
                              param.input_dim(top_id * 4 + 2),
                              param.input_dim(top_id * 4 + 3));
      } else {
        blob_pointer->Reshape(param.input_shape(top_id));
      }
      net_input_blob_indices_.push_back(blob_id);
      net_input_blobs_.push_back(blob_pointer.get());
    } else {
      top_id_vecs_[layer_id].push_back(blob_id);  // 更新当前层的top_id_vecs_
      top_vecs_[layer_id].push_back(blob_pointer.get()); // 用当前top blob的指针更新数据成员
    }
  }
 // if(MY_DEBUG)
   // LOG(INFO) << "available_blobs: " << available_blobs.isEmpty();
  if (available_blobs) { available_blobs->insert(blob_name); }
  
  if(MY_DEBUG)
    LOG(INFO) << "AppendTop()调用结束"; 
} // end of AppendTop

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  if(MY_DEBUG)
    LOG(INFO) << "进入AppendBottom()函数";
  //获取当前层的LayerParameter类的对象layer_param 
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  //blob_name_to_idx是一个map,其关键字是不重复的。blob_name_to_idx在输入层初始化过了-->*blob_name_to_idx)[blob_name] = blob_id
  const int blob_id = (*blob_name_to_idx)[blob_name];
  // 输出blob的流动方向
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
  //调用shared_ptr类的get()方法提取存储在blobs_中的中间变量
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());  // 更新数据成员bottom_vecs_
  bottom_id_vecs_[layer_id].push_back(blob_id); // 更新数据成员bottom_id_vecs_
  available_blobs->erase(blob_name); // 当前blob_name的blob已经完成了使命，所以需要擦除？？？？？
  bool propagate_down = true; // 默认需要反馈
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0)
    propagate_down = layer_param.propagate_down(bottom_id);
  const bool need_backward = blob_need_backward_[blob_id] &&
                          propagate_down;
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  if(MY_DEBUG)
    LOG(INFO) << "调用Net:AppendParam()";
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    // 名字为空时直接压如其param_id
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  if(MY_DEBUG)
    LOG(INFO) << "param_display_names_.back(): " <<param_display_names_.back();  
  if(MY_DEBUG)
    LOG(INFO) << "params_.size():  " <<params_.size();  
  // Append参数blob没一次循环，net_param_id和param_id_vecs_都会更新
  const int net_param_id = params_.size();// 整个网络的参数的id，不管这个参数有没有non-empty name，是否参与share,比如：第一次调用conv1时，其值为0,当第二次调用conv1时，其值加1
  // 更新各个成员数据
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  // 将整个网络的参数按层的形式来存储，存储的元素可以理解为params_这个向量的下标值
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  // 获取每个param_id所对应的ParamSpec类型的成员
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    // 如果param_name不为空，而且能够在param_names_index_中找到，说明这个parameter已经存在于之前的某个或者某些网络层里，说明这个parameter是共享于多个layer
    // 在caffe.proto的message ParamSpec里关于name的注释——>To share a parameter between two layers, give it a (non-empty) name, 可见，如果一个parameter是共享与多个网络层，那么它会有一个非空的name
    param_owners_.push_back(-1); // -1表示当前Layer就是该parameter的"owner"
    // 添加param_name
    if (param_name.size()) {
      //map<string, int> param_names_index_是整个网络的参数non-empty name与index的映射。
      //注意，这个name是ParamSpec 类型中的name,而且，""To share a parameter between two layers, give it a (non-empty) name"",所以说这个map中存储的pair是<会被share的parameter_name, 其对应index>
      param_names_index_[param_name] = net_param_id; //虽然每一次循环，net_param_id都会更新，但是net_param_id只有当param_name.size()>0时才会被压入向量param_names_index_
    }
    const int learnable_param_id = learnable_params_.size();
    // 在模板类layer中，定义了一个blobs_成员，其存储的就是learnable parameter。随后压入learnable_param_id
    learnable_params_.push_back(params_[net_param_id].get());
    if(MY_DEBUG)
      LOG(INFO) << "learnable_params_.size(): " << learnable_params_.size();
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
} // end of AppendParam

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  if (debug_info_) {
    for (int i = 0; i < net_input_blobs_.size(); ++i) {
      InputDebugInfo(i);
    }
  }
  for (int i = start; i <= end; ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

// 用于前馈预先填满，即预先进行一次前馈
template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::ForwardPrefilled(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

// 把网络输入层的blob读到net_input_blobs_,然后进行反馈，计算出loss
template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  // Copy bottom to internal bottom
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return ForwardPrefilled(loss);
}

// Forward的重载，只是输出层的blob以string的格式传入
template <typename Dtype>
string Net<Dtype>::Forward(const string& input_blob_protos, Dtype* loss) {
  BlobProtoVector blob_proto_vec;
  if (net_input_blobs_.size()) {
    blob_proto_vec.ParseFromString(input_blob_protos);
    CHECK_EQ(blob_proto_vec.blobs_size(), net_input_blobs_.size())
        << "Incorrect input size.";
    for (int i = 0; i < blob_proto_vec.blobs_size(); ++i) {
      net_input_blobs_[i]->FromProto(blob_proto_vec.blobs(i));
    }
  }
  ForwardPrefilled(loss);
  blob_proto_vec.Clear();
  for (int i = 0; i < net_output_blobs_.size(); ++i) {
    net_output_blobs_[i]->ToProto(blob_proto_vec.add_blobs());
  }
  string output;
  blob_proto_vec.SerializeToString(&output);
  return output;
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
  }
}

template <typename Dtype>
void Net<Dtype>::InputDebugInfo(const int input_id) {
  const Blob<Dtype>& blob = *net_input_blobs_[input_id];
  const string& blob_name = blob_names_[net_input_blob_indices_[input_id]];
  const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
  LOG_IF(INFO, Caffe::root_solver())
      << "    [Forward] "
      << "Input " << blob_name << " data: " << data_abs_val_mean;
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

// 对整个网络进行反向传播
template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  for (int i = 0; i < net_input_blob_indices_.size(); ++i) {
    param->add_input(blob_names_[net_input_blob_indices_[i]]);
  }
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

// 更新params_中blob的值
template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

// 清空上一次所有参数的梯度
template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
