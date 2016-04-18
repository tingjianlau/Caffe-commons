#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

//主要工作是：Reshape top blob 和 prefetch得到的batch的data_ blob, label_ blob
template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if(MY_DEBUG)
    LOG(INFO) << "调用了DataLayer类的DataLayerSetUp函数";
  // 获取参数列表中的batch_size的大小
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  // 读取一个数据点，用来初始化top blob。所谓的初始化，主要是指Reshape,Datum是一个由caffe.proto定义的一个模板类
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum. data_transformer是一个由基类定义的一个指向DataTransformer类型的智能指针数据成员变量。得到的是第一张图片的基本shape信息，其中shape[0]==1,因为此时图片数量只有一张。
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  // transformed_data是直接基类定义的一个Blob类型的变量，调用其成员函数Reshape函数，重置数据成员count_,shape_等
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size; // 将批量处理的数据大小设为top_shape的num
  top[0]->Reshape(top_shape); // 取传入的top的Vector的第一个blob"定形"
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    // prefetch[]是直接基类声明的一个Batch类型的对象成员,包括data_和label_两个公有数据成员
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  // 打印当前blo top的shape
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label output_labels是其基类BaseDataLayer中定义的布尔型数据成员,在其成员函数LayerSetUp中更新值,若top.size()!=1,则为true
  if (this->output_labels_) {
    if(MY_DEBUG)
      LOG(INFO) << "output_labels: " << this->output_labels_;
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape); // 更新label对应的blob的shape
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size; // 将每次批量处理的大小设为当前blob的num
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
