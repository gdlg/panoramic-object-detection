#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/ring_pad_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RingPadBackward(
    Dtype* bottom_diff, const Dtype* top_diff, const int outer_dim, const int bottom_inner_dim, const int top_inner_dim, const int pad) {

  CUDA_KERNEL_LOOP(index, outer_dim) {
    Dtype* bottom_base = bottom_diff + index * bottom_inner_dim;
    const Dtype* top_base = top_diff + index * top_inner_dim;

    for (int k = 0; k < pad; ++k) {
        bottom_base[k] = top_base[k+pad] + top_base[k + top_inner_dim - pad];
        bottom_base[k+bottom_inner_dim-pad] = top_base[k] + top_base[k + top_inner_dim - 2*pad];
    }
  }
}

template <typename Dtype>
void RingPadLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int outer_dim = bottom[0]->count(0, axis_);
  int bottom_inner_dim = bottom[0]->count(axis_, bottom[0]->num_axes());
  int top_inner_dim = top[0]->count(axis_, bottom[0]->num_axes());
  int pad = padding_ * bottom[0]->count(axis_+1, bottom[0]->num_axes());

  cudaMemcpy2D(top_data + top_inner_dim - pad, top_inner_dim * sizeof(Dtype),
               bottom_data, bottom_inner_dim * sizeof(Dtype),
               pad * sizeof(Dtype),
               outer_dim,
               cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(top_data, top_inner_dim * sizeof(Dtype),
               bottom_data + bottom_inner_dim - pad, bottom_inner_dim * sizeof(Dtype),
               pad * sizeof(Dtype),
               outer_dim,
               cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(top_data + pad, top_inner_dim * sizeof(Dtype),
               bottom_data, bottom_inner_dim * sizeof(Dtype),
               bottom_inner_dim * sizeof(Dtype),
               outer_dim,
               cudaMemcpyDeviceToDevice);
}

template <typename Dtype>
void RingPadLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    int outer_dim = bottom[0]->count(0, axis_);
    int bottom_inner_dim = bottom[0]->count(axis_, bottom[0]->num_axes());
    int top_inner_dim = bottom[0]->count(axis_, bottom[0]->num_axes());
    int pad = padding_ * bottom[0]->count(axis_+1, bottom[0]->num_axes());

    cudaMemcpy2D(bottom_diff + pad, bottom_inner_dim * sizeof(Dtype),
                 top_diff + 2*pad, top_inner_dim * sizeof(Dtype),
                 (bottom_inner_dim - 2*pad) * sizeof(Dtype),
                 outer_dim,
                 cudaMemcpyDeviceToDevice);

    RingPadBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(outer_dim), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_diff, top_diff, outer_dim, bottom_inner_dim, top_inner_dim, pad);
    for (int n = 0; n < outer_dim; ++n) {
      Dtype* bottom_base = bottom_diff + n * bottom_inner_dim;
      const Dtype* top_base = top_diff + n * top_inner_dim;

      for (int k = 0; k < pad; ++k) {
          bottom_base[k] = top_base[k+pad] + top_base[k + top_inner_dim - pad];
          bottom_base[k+bottom_inner_dim-pad] = top_base[k] + top_base[k + top_inner_dim - 2*pad];
      }

    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RingPadLayer);

}  // namespace caffe
