#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

//  构造函数，初始化两个Net类，net和test_net，并调用init()
//  输入为SolverParameter类型的param
template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  if(MY_DEBUG)
    LOG(INFO) << "调用Solver构造函数成功";
  Init(param);
}

//  构造函数，初始化两个Net类，net和test_net，并调用init()
//  输入为string类型的param_file
template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

// 功能：初始化网络
template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG(INFO) << "调用Init()";
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  if(MY_DEBUG)
    LOG(INFO) << "param.DebugString()输出结束";
  param_ = param; // 为数据成员赋值
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  // 步骤1：设置随机数种子
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    // 调用Caffe命名空间里的set_random_seed函数，而不是caffe类的该
    // 函数，param_.random_seed()实际上调用的是::google::protobuf::int64 random_seed()
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet(); // 步骤2：初始化训练网络
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  if(MY_DEBUG)
    LOG(INFO) << "InitTrainNet()开始...";

  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  if(MY_DEBUG)
    LOG(INFO) << "num_train_nets的值" << num_train_nets; 
  const string& field_names = "net, net_param, train_net, train_net_param";
  // 只能有一个train net
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  // 当前实验只构造了net网
  if (param_.has_net()) {
    // 输出net protocol buffer定义所在的文件
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    // 读取net_param
    if(MY_DEBUG)
      LOG(INFO) << "读取net_param";
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  // 声明一个Google buffer的NetState对象，并设置
  NetState net_state;
  net_state.set_phase(TRAIN); // 设置当前阶段是TRAIN
  // 从低到高获取state，最终从最高优先级SolverParameter类型种的train_state,显然这会覆盖掉之前获取的state
  net_state.MergeFrom(net_param.state()); 
  //if(MY_DEBUG)
   // LOG(INFO) << "param_.train_state() phase:" << param_.train_state().phase << "level: "  << param_.train_state().level << "stage: " << param.train_state().stage;
  // 这里获取的state可以为NetParameter的state赋值，然后可以根据LayerParameter中的include和exclude来确定该层是否应该包含在网络中
  net_state.MergeFrom(param_.train_state()); 
  // 返回一个NetState对象，这是Initialize train net的一部分工作，InitTestNets也是如此
  net_param.mutable_state()->CopyFrom(net_state); //将当前已经“收集”到的net_state赋给可变的state
  // 申请一块Net空间
  if (Caffe::root_solver()) {
    if(MY_DEBUG)
      LOG(INFO) << "调用Net构造函数开始";
    //调用模板类的构造函数，进行Net的初始化
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  if(MY_DEBUG)
	LOG(INFO) << "调用InitTestNets()";
  CHECK(Caffe::root_solver());
  // 通过proto文件获取相关参数
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if(MY_DEBUG)
	LOG(INFO) << "has_net_param: " << has_net_param << " has_net_file: "
	<< has_net_file << "  num_test_net_params: " <<  num_test_net_params << " num_test_net_files: " << num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  vector<Blob<Dtype>*> bottom_vec;
  // 设置开始的迭代次数（如果是从之前的snapshot恢复的，那iter_等于其迭代次数)和结束的迭代次数
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  // 输出的loss为前average_loss次loss的平均值，在solver.proto中设置，默认1
  int average_loss = this->param_.average_loss();
  // losses存储之前的average_loss个loss，smoothed_loss为最后要输出的均值
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;

  while (iter_ < stop_iter) {
    // zero-init the params
	// 清空上一次所有参数的梯度
    net_->ClearParamDiffs();
	// 判断是否需要测试
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
	  // 此时用到的网络时测试网络，即test_net_
      TestAll();
	  // 判断是否需要提前结束迭代
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
	// 判断当前迭代次数是否需要显示loss等信息
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
	// iter_size也是在solver.prototxt里设置，实际上的batch_size=iter_size*网络定义里的batch_size
	// 因此每一次迭代的loss是iter_size次迭代的和，再除以iter_size,这个loss是通过调用Net::ForwardBackward计算的
	// 在GPU显存不够时使用，例如如果把batch_size设置为128，但是会out_of_memory，借助这个方法，可以设置batch_size=32,iter_size=4,那实际上每次迭代还是处理了128个数据
    for (int i = 0; i < param_.iter_size(); ++i) {
	  // 此时用到的网络是训练网络，即net_
      loss += net_->ForwardBackward(bottom_vec);
    }
    loss /= param_.iter_size();
	// 计算要输出的smoothed_loss，如果losses里还没有存够average个loss则将当前的loss插入，如果已经已经存够了，则将之前的替换掉
    // average the loss across iterations for smoothed reporting
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
	// 输出当前迭代的信息
    if (display) {
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", loss = " << smoothed_loss;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
	// 执行梯度的更新，这个函数在基类Solver中没有实现，会调用每个子类自己的实现
    ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

	// 调用GetRequestedAction，实际是通过action_request_function_函数指针调用之前设置好
	// signal_hanlder的CheckForSignals函数，这个函数的作用是
	// 会根据之前是否遇到系统信号以及信号的类型和我们设置或默认的方式返回处理的方式
    SolverAction::Enum request = GetRequestedAction();

	// 判断当前迭代是否需要snapshot，如果request等于SNAPSHOT，则也需要
    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
	// 如果request是STOP，则修改，之后就会提前结束迭代
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  } // end of while
} // end of Step

// 对整个网络进行训练（也就是运行caffe训练某个模型)的时候，实际上是在运行caffe.cpp中的train()函数，而这个函数实际上是实例化一个Solver对象，初始化后调用了Solver中的Solve()方法。调用此方法训练网络，其中调用Step()来迭代，迭代param_.max_iter()-iter_次. 
template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  if(MY_TEST_DEBUG)
	LOG(INFO) << "进入Solver::Slove()";
  // 检查当前是否是root_slover（多GPU模式下，只有root_slover才运行这一部分的代码)
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.一开始被赋值为false，也就是现在没有要求在优化结束前推出
  requested_early_exit_ = false;

  // 判断这个指针是否NULL，如果不是则需要从resume_file存储的路径里读取之前训练的solver status
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // 然后调用了Step函数，执行了实际的逐步的迭代过程
  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  Step(param_.max_iter() - iter_);
  // 迭代结束或遇到系统信号提前结束后，判断是否需要在训练结束之后snapshot
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  // 如果在Step函数的迭代过程中遇到了系统信号，且我们的处理方式设置为"STOP"
  // 那么其会被修改为true，迭代提前结束，输出相关信息
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  // 判断是否需要输出最后的loss
  if (param_.display() && iter_ % param_.display() == 0) {
    Dtype loss;
	// 最后在已经训练好的网络中计算loss
    net_->ForwardPrefilled(&loss);
    if(MY_TEST_DEBUG)
	    LOG(INFO) << "输出最后的loss";
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  }
  if(MY_TEST_DEBUG){
	LOG(INFO) << "net_input_blobs_.size(): " << net_->num_inputs() << " " << test_nets_[0]->num_inputs();
	LOG(INFO) << "训练网络的层数: " << net_->layers().size();
	LOG(INFO) << "测试网络的层数: " << test_nets_[0]->layers().size();
	LOG(INFO) << "训练网络的可学习参数的个数: " << net_->learnable_params().size();
	LOG(INFO) << "测试网络的可学习参数的个数: " << test_nets_[0]->learnable_params().size();
  }
  // 判断是否需要最后Test
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
	if(MY_TEST_DEBUG)
		LOG(INFO) << "进入最后的Test";
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

// 测试整个test_net_，并输出相应的测试信息，如loss等
template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  // 打印测试网络信息
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score; // 装载测试网络输出的loss和accuracy的累加的数值
  vector<int> test_score_output_id; // 相应的id
  vector<Blob<Dtype>*> bottom_vec;
  // test_nets_是已经初始化后的得到的训练网络
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  // test_iter specifies how many forward passed the test should carry out
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
	// call the Forward() to cumpute the loss and accuracy
	// and result the Net::net_output_blobs_
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
	// default faslse
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
	  // result.size()=2, result[j]=1，即只有Accuracy层的top[0]->count()
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  // default fasle
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  if(false && MY_TEST_DEBUG){
	// 两个都输出2
	LOG(INFO) << "test_score.size(): " << test_score.size();
	LOG(INFO) << "test_score_output_id.size(): " << test_score_output_id.size();
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
