// ------------------------------------------------------------------
// MS-CNN
// Copyright (c) 2016 The Regents of the University of California
// see mscnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifndef CAFFE_PROPOSAL_TARGET_LAYERS_HPP_
#define CAFFE_PROPOSAL_TARGET_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <Eigen/StdVector>
typedef std::vector<Eigen::Vector4f,Eigen::aligned_allocator<Eigen::Vector4f>> VecOfVector4f;
typedef std::vector<Eigen::Vector3f,Eigen::aligned_allocator<Eigen::Vector3f>> VecOfVector3f;
typedef std::vector<Eigen::Matrix3f,Eigen::aligned_allocator<Eigen::Matrix3f>> VecOfMatrix3f;
typedef std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> VecOfMatrix4f;

namespace caffe {

template <typename Dtype>
class ProposalTargetLayer : public Layer<Dtype> {
 public:
  explicit ProposalTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ProposalTarget"; }

  virtual inline int ExactBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 5; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);*/
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  /*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }*/
  
  shared_ptr<Caffe::RNG> shuffle_rng_;
  int batch_size_;
  int cls_num_;
  bool has_sample_weight_;
  bool has_3d_boxes_and_proj_mat_input_;
  bool output_3d_boxes_;

  struct WeightedFeatureIndex {
      int data;
      int inside_weights;
      int outside_weights;
  };

  struct WeightedFeatureData {
      Dtype* data = nullptr;
      Dtype* inside_weights = nullptr;
      Dtype* outside_weights = nullptr;
  };

  struct FrameInfo {
      WeightedFeatureIndex location_targets_idx_, distance_targets_idx_, orientation_targets_idx_;
      int quadrant_targets_idx_;
  };

  struct Frame {
      VecOfVector3f gt_location;
      vector<float> gt_orientation;

      vector<vector<Dtype>> location_targets, distance_targets, orientation_targets, quadrant_targets;
  };

  struct FrameData {
      WeightedFeatureData location, distance, orientation;
      Dtype* quadrant = nullptr;
  };

  struct FrameObject {
    vector<Dtype> location_target = {0,0};
    vector<Dtype> distance_target = {0};
    vector<Dtype> quadrant_target = {0};
    vector<Dtype> orientation_target = {0,0};
  };

  WeightedFeatureIndex bbox_targets_idx_, location_targets_idx_, distance_targets_idx_, orientation_targets_idx_, size_targets_idx_;
  int sampled_rois_idx_, labels_idx_, matched_gt_boxes_idx_, sample_weights_idx_, roi_height_idx_;

  bool use_panoramic_;

  FrameInfo current_frame_info;

  void set_weighted_feature_idx(WeightedFeatureIndex &feature, int &index);
  void resize_weighted_feature_top(const vector<Blob<Dtype>*>& top, WeightedFeatureIndex feature_idx, int batch_size, int channels_num, int width=1, int height=1);
  WeightedFeatureData get_weighted_feature_top_data(const vector<Blob<Dtype>*>& top, WeightedFeatureIndex feature_idx, int batch_size, int channels_num, int width=1, int height=1);
  template<class U>
  void fill_weighted_data(WeightedFeatureData dst, const U& src, int dst_start, int num);

  void set_frame_idx(FrameInfo& frame, int &index);
  void resize_frame_top(FrameInfo& frame, const vector<Blob<Dtype>*>& top, int batch_size);
  FrameData get_frame_top_data(FrameInfo& frame, const vector<Blob<Dtype>*>& top, int batch_size);
  void set_frame_object(const Frame& frame, const Eigen::Matrix3f &P, const Eigen::Matrix4f &M, const vector<float> bbox_means, const vector<float> bbox_stds, int gtid, Dtype bb_ctr_x, Dtype bb_ctr_y, Dtype bb_width, Dtype bb_height, FrameObject& obj);
  void append_frame_object(Frame &frame, const FrameObject& obj);
};

}  // namespace caffe

#endif  // CAFFE_PROPOSAL_TARGET_LAYERS_HPP_
