#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/ring_pad_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RingPadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const RingPadParameter& param = this->layer_param_.ring_pad_param();
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  padding_ = param.pad();
}

template <typename Dtype>
void RingPadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  shape[axis_] += 2*padding_;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void RingPadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int outer_dim = bottom[0]->count(0, axis_);
  int bottom_inner_dim = bottom[0]->count(axis_, bottom[0]->num_axes());
  int top_inner_dim = bottom[0]->count(axis_, bottom[0]->num_axes());
  int pad = padding_ * bottom[0]->count(axis_+1, bottom[0]->num_axes());
  for (int n = 0; n < outer_dim; ++n) {
    const Dtype* bottom_base = bottom_data + n * bottom_inner_dim;
    Dtype* top_base = top_data + n * top_inner_dim;
    std::copy_n(bottom_base, pad, top_base + top_inner_dim - pad);
    std::copy_n(bottom_base + bottom_inner_dim - pad, pad, top_base);
    std::copy_n(bottom_base, bottom_inner_dim, top_base + pad);
  }
}

template <typename Dtype>
void RingPadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    int outer_dim = bottom[0]->count(0, axis_);
    int bottom_inner_dim = bottom[0]->count(axis_, bottom[0]->num_axes());
    int top_inner_dim = bottom[0]->count(axis_, bottom[0]->num_axes());
    int pad = padding_ * bottom[0]->count(axis_+1, bottom[0]->num_axes());
    for (int n = 0; n < outer_dim; ++n) {
      Dtype* bottom_base = bottom_diff + n * bottom_inner_dim;
      const Dtype* top_base = top_diff + n * top_inner_dim;

      for (int k = 0; k < pad; ++k) {
          bottom_base[k] = top_base[k+pad] + top_base[k + top_inner_dim - pad];
          bottom_base[k+bottom_inner_dim-pad] = top_base[k] + top_base[k + top_inner_dim - 2*pad];
      }

      std::copy_n(top_base + 2*pad, bottom_inner_dim - 2*pad, bottom_base + pad);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RingPadLayer);
#endif

INSTANTIATE_CLASS(RingPadLayer);
REGISTER_LAYER_CLASS(RingPad);

}  // namespace caffe
