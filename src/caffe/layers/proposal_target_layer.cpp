// ------------------------------------------------------------------
// MS-CNN
// Copyright (c) 2016 The Regents of the University of California
// see mscnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layers/proposal_target_layer.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace caffe {
    
template <typename Dtype>
void ProposalTargetLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ProposalTargetParameter proposal_target_param = this->layer_param_.proposal_target_param();
  cls_num_ = proposal_target_param.cls_num();
  batch_size_ = this->layer_param_.proposal_target_param().batch_size();
  has_3d_boxes_and_proj_mat_input_ = this->layer_param_.proposal_target_param().has_3d_boxes_and_proj_mat_input();
  output_3d_boxes_ = this->layer_param_.proposal_target_param().output_3d_boxes();
  has_sample_weight_ = this->layer_param_.proposal_target_param().output_sample_weight();
  use_panoramic_ = this->layer_param_.proposal_target_param().use_panoramic();
  
  const unsigned int shuffle_rng_seed = caffe_rng_rand();
  shuffle_rng_.reset(new Caffe::RNG(shuffle_rng_seed));
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::set_weighted_feature_idx(WeightedFeatureIndex &feature, int &index)
{
    feature.data = index++;
    feature.inside_weights = index++;
    feature.outside_weights = index++;
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::resize_weighted_feature_top(const vector<Blob<Dtype>*>& top, WeightedFeatureIndex feature_idx, int batch_size, int channels_num, int width, int height)
{
  top[feature_idx.data]->Reshape(batch_size, channels_num, width, height);
  top[feature_idx.inside_weights]->Reshape(batch_size, channels_num, width, height);
  top[feature_idx.outside_weights]->Reshape(batch_size, channels_num, width, height);
}

template<class Dtype>
typename ProposalTargetLayer<Dtype>::WeightedFeatureData ProposalTargetLayer<Dtype>::get_weighted_feature_top_data(const vector<Blob<Dtype>*>& top, WeightedFeatureIndex feature_idx, int batch_size, int channels_num, int width, int height)
{
    resize_weighted_feature_top(top, feature_idx, batch_size, channels_num, width, height);

    WeightedFeatureData data;
    data.data = top[feature_idx.data]->mutable_cpu_data();
    caffe_set(top[feature_idx.data]->count(), Dtype(0), data.data);

    data.inside_weights = top[feature_idx.inside_weights]->mutable_cpu_data();
    caffe_set(top[feature_idx.inside_weights]->count(), Dtype(0), data.inside_weights);

    data.outside_weights = top[feature_idx.outside_weights]->mutable_cpu_data();
    caffe_set(top[feature_idx.outside_weights]->count(), Dtype(0), data.outside_weights);

    return data;
}

template<class Dtype> template<class U>
void ProposalTargetLayer<Dtype>::fill_weighted_data(WeightedFeatureData dst, const U& src, int dst_start, int num)
{
    for (int i = 0; i < num; ++i)
    {
      dst.data[dst_start+i] = src[i];
      dst.inside_weights[dst_start+i] = 1; //every dim has the same weight
      dst.outside_weights[dst_start+i] = 1;
    }
}

template<class Dtype>
void ProposalTargetLayer<Dtype>::set_frame_idx(FrameInfo& frame, int &index)
{
  set_weighted_feature_idx(frame.location_targets_idx_, index);
  set_weighted_feature_idx(frame.distance_targets_idx_, index);
  frame.quadrant_targets_idx_ = index++;
  set_weighted_feature_idx(frame.orientation_targets_idx_, index);
}

template<class Dtype>
void ProposalTargetLayer<Dtype>::resize_frame_top(FrameInfo& frame, const vector<Blob<Dtype>*>& top, int batch_size)
{
  resize_weighted_feature_top(top, frame.location_targets_idx_, batch_size, cls_num_*2);
  resize_weighted_feature_top(top, frame.distance_targets_idx_, batch_size, cls_num_);
  top[frame.quadrant_targets_idx_]->Reshape(batch_size, 1, 1, 1);
  resize_weighted_feature_top(top, frame.orientation_targets_idx_, batch_size, cls_num_, 2);
}

template <typename Dtype>
auto ProposalTargetLayer<Dtype>::get_frame_top_data(FrameInfo& frame, const vector<Blob<Dtype>*>& top, int batch_size) -> FrameData
{
    FrameData data;
    data.location = get_weighted_feature_top_data(top, frame.location_targets_idx_, batch_size, 2*cls_num_);
    data.distance = get_weighted_feature_top_data(top, frame.distance_targets_idx_, batch_size, cls_num_);
    top[frame.quadrant_targets_idx_]->Reshape(batch_size, 1, 1, 1);
    data.quadrant = top[frame.quadrant_targets_idx_]->mutable_cpu_data();
    data.orientation = get_weighted_feature_top_data(top, frame.orientation_targets_idx_, batch_size, cls_num_, 2);
    return data;
}

Eigen::Vector2f spherical_project(const Eigen::Matrix3f &P, const Eigen::Matrix4f &M, const Eigen::Vector3f p) {
    const Eigen::Vector4f p2 = M * p.homogeneous();

    float lon = atan2(p2(0), p2(2));
    float lat = asin(p2(1) / p2.norm());

    return (P * Eigen::Vector3f{lon,lat,1}).eval().hnormalized();
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::set_frame_object(const Frame& frame, const Eigen::Matrix3f &P, const Eigen::Matrix4f &M, const vector<float> bbox_means, const vector<float> bbox_stds, int gtid, Dtype bb_ctr_x, Dtype bb_ctr_y, Dtype bb_width, Dtype bb_height, FrameObject& obj) {

  float focal_length;


  Dtype gt_ctr_3d_x, gt_ctr_3d_y;

  if (use_panoramic_) {
    auto gt_ctr_3d = spherical_project(P, M, frame.gt_location[gtid]);
    gt_ctr_3d_x = gt_ctr_3d(0);
    gt_ctr_3d_y = gt_ctr_3d(1);
    focal_length = fabs(P(0,0));
  } else {
    Eigen::Matrix4f P4x4 = Eigen::Matrix4f::Identity();
    P4x4.block<3,3>(0,0) = P;
    P4x4 *= M;
    Eigen::Vector4f hxyz = P4x4 * frame.gt_location[gtid].homogeneous();
    gt_ctr_3d_x = hxyz(0)/hxyz(2);
    gt_ctr_3d_y = hxyz(1)/hxyz(2);
    focal_length = fabs(P4x4(0,0));
  }

  Dtype targets_dx_3d = (gt_ctr_3d_x - bb_ctr_x) / bb_width;
  Dtype targets_dy_3d = (gt_ctr_3d_y - bb_ctr_y) / bb_height;

  targets_dx_3d = (targets_dx_3d - bbox_means[0])/bbox_stds[0];
  targets_dy_3d = (targets_dy_3d - bbox_means[1])/bbox_stds[1];

  obj.location_target[0] = targets_dx_3d;
  obj.location_target[1] = targets_dy_3d;

  // Compute the distance to the target. It’s better to do it in image space
  // therefore we add the last row of the projection matrix
  auto image_frame_location = (M * frame.gt_location[gtid].homogeneous()).eval().hnormalized();
  obj.distance_target[0] = focal_length/image_frame_location.norm();

  float theta = frame.gt_orientation[gtid] - atan2(frame.gt_location[gtid](0), frame.gt_location[gtid](2));
  float cos_theta = cos(theta);
  float sin_theta = sin(theta);
  obj.quadrant_target[0] = 2*(cos_theta > 0) + (sin_theta > 0);
  obj.orientation_target[0] = cos_theta*cos_theta;
  obj.orientation_target[1] = sin_theta*sin_theta;
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::append_frame_object(Frame &frame, const FrameObject& obj)
{
  frame.location_targets.push_back(obj.location_target);
  frame.distance_targets.push_back(obj.distance_target);
  frame.quadrant_targets.push_back(obj.quadrant_target);
  frame.orientation_targets.push_back(obj.orientation_target);
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int index = 0;
  sampled_rois_idx_ = index++;
  labels_idx_ = index++;
  set_weighted_feature_idx(bbox_targets_idx_, index);
  set_weighted_feature_idx(size_targets_idx_, index);
  set_frame_idx(current_frame_info, index);
  matched_gt_boxes_idx_ = index++;
  if (has_sample_weight_) {
      sample_weights_idx_ = index++;
  }

  roi_height_idx_ = index++;

  //sampled rois (img_id, x1, y1, x2, y2)
  top[sampled_rois_idx_]->Reshape(batch_size_, 5, 1, 1);
  //labels
  top[labels_idx_]->Reshape(batch_size_, 1, 1, 1);
  //bbox targets
  resize_weighted_feature_top(top, bbox_targets_idx_, batch_size_, cls_num_*4);
  if (output_3d_boxes_) {
    resize_weighted_feature_top(top, size_targets_idx_, batch_size_, cls_num_*3);
    resize_frame_top(current_frame_info, top, batch_size_);
  }

  //matched gt boxes for bbox evaluation (label, x1, y1, x2, y2, overlap)
  top[matched_gt_boxes_idx_]->Reshape(batch_size_, 6, 1, 1);
  //sample weights
  if (has_sample_weight_) {
    top[sample_weights_idx_]->Reshape(batch_size_, 1, 1, 1);
  }
  CHECK_EQ(bottom[0]->channels(),5);
  int expected_num_channels = 7;
  if (has_3d_boxes_and_proj_mat_input_) expected_num_channels += 7;
  CHECK_EQ(bottom[1]->channels(), expected_num_channels);
  if (has_3d_boxes_and_proj_mat_input_) {
    CHECK_EQ(bottom[2]->channels(),6);
    CHECK_EQ(bottom[3]->channels(),16);
  }

  top[roi_height_idx_]->Reshape({batch_size_, cls_num_});
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //parameters
  const float fg_fraction = this->layer_param_.proposal_target_param().fg_fraction();
  int fg_rois_per_batch = round(fg_fraction*batch_size_);
  const int num_img_per_batch = this->layer_param_.proposal_target_param().num_img_per_batch();
  const float fg_thr = this->layer_param_.proposal_target_param().fg_thr();
  const float bg_thr_hg = this->layer_param_.proposal_target_param().bg_thr_hg();
  const float bg_thr_lw = this->layer_param_.proposal_target_param().bg_thr_lw();
  CHECK_GT(fg_thr,bg_thr_hg);
  const int img_width = this->layer_param_.proposal_target_param().img_width();
  const int img_height = this->layer_param_.proposal_target_param().img_height();
  const bool iou_weighted = this->layer_param_.proposal_target_param().iou_weighted();
  
  // bbox mean and std
  bool do_bbox_norm = false;
  vector<float> bbox_means, bbox_stds;
  if (this->layer_param_.bbox_reg_param().bbox_mean_size() > 0
      && this->layer_param_.bbox_reg_param().bbox_std_size() > 0) {
    do_bbox_norm = true;
    int num_bbox_means = this->layer_param_.bbox_reg_param().bbox_mean_size();
    int num_bbox_stds = this->layer_param_.bbox_reg_param().bbox_std_size();
    CHECK_EQ(num_bbox_means,4); CHECK_EQ(num_bbox_stds,4);
    for (int i = 0; i < 4; i++) {
      bbox_means.push_back(this->layer_param_.bbox_reg_param().bbox_mean(i));
      bbox_stds.push_back(this->layer_param_.bbox_reg_param().bbox_std(i));
    }
  }
  
  //inputs
  int num_rois = bottom[0]->num();
  int num_gt_boxes = bottom[1]->num();
  //[img_id, x1, y1, x2, y2]
  const Dtype* rois_boxes_data = bottom[0]->cpu_data();
  const int rois_dim = bottom[0]->channels();
  //[img_id, x1, y1, x2, y2, label, ignored]
  const Dtype* gt_boxes_data = bottom[1]->cpu_data();
  const int gt_dim = bottom[1]->channels();

  //[img_id, matrix]
  VecOfMatrix3f proj_mat(num_img_per_batch);
  VecOfMatrix4f model_mat(num_img_per_batch);

  if (has_3d_boxes_and_proj_mat_input_) {
    const Dtype* proj_mat_data = bottom[2]->cpu_data();
    CHECK_EQ(bottom[2]->num(), num_img_per_batch);
    CHECK_EQ(bottom[2]->channels(), 6);

    for (int k = 0; k < num_img_per_batch; ++k) {
      for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j) {
        proj_mat[k](i,j) = proj_mat_data[6*k + i*3 + j];
      }
      proj_mat[k](2,0) = 0; proj_mat[k](2,1) = 0; proj_mat[k](2,2) = 1;
    }

    const Dtype* model_mat_data = bottom[3]->cpu_data();
    CHECK_EQ(bottom[3]->num(), num_img_per_batch);
    CHECK_EQ(bottom[3]->channels(), 16);

    for (int k = 0; k < num_img_per_batch; ++k) {
      for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        model_mat[k](i,j) = model_mat_data[16*k + i*4 + j];
      }
    }
  }
  
  //[img_id x1 y1 w h]
  vector<vector<Dtype> > rois_boxes, gt_boxes;
  for (int i = 0; i < num_rois; i++) {
    vector<Dtype> bb(5);
    Dtype score = rois_boxes_data[i*rois_dim];
    Dtype x1 = rois_boxes_data[i*rois_dim+1];
    Dtype y1 = rois_boxes_data[i*rois_dim+2];
    Dtype x2 = rois_boxes_data[i*rois_dim+3];
    Dtype y2 = rois_boxes_data[i*rois_dim+4];

    bb[0] = score;
    bb[1] = x1;
    bb[2] = y1;
    bb[3] = x2-x1+1;
    bb[4] = y2-y1+1;

    CHECK_LT(bb[0],num_img_per_batch);
    rois_boxes.push_back(bb);
  }
  //append gt boxes to the end of rois
  vector<Dtype> gt_labels, gt_ignored;
  VecOfVector3f gt_size;
  Frame current_frame;
  for (int i = 0; i < num_gt_boxes; i++) {
    vector<Dtype> bb(5);

    Dtype score = gt_boxes_data[i*gt_dim];
    Dtype x1 = gt_boxes_data[i*gt_dim+1];
    Dtype y1 = gt_boxes_data[i*gt_dim+2];
    Dtype x2 = gt_boxes_data[i*gt_dim+3];
    Dtype y2 = gt_boxes_data[i*gt_dim+4];

    bb[0] = score;
    bb[1] = x1;
    bb[2] = y1;
    bb[3] = x2-x1+1;
    bb[4] = y2-y1+1;

    CHECK_LT(bb[0],num_img_per_batch);

    Eigen::Vector3f size(
            gt_boxes_data[i*gt_dim+7],
            gt_boxes_data[i*gt_dim+8],
            gt_boxes_data[i*gt_dim+9]);
    current_frame.gt_location.push_back({
            (float)gt_boxes_data[i*gt_dim+10],
            (float)gt_boxes_data[i*gt_dim+11],
            (float)gt_boxes_data[i*gt_dim+12]});
    current_frame.gt_orientation.push_back(gt_boxes_data[i*gt_dim+13]);
    gt_size.push_back(size);

    if (bb[3] != 0) {
        rois_boxes.push_back(bb);
        ++num_rois;
    }
    gt_boxes.push_back(bb);
    gt_labels.push_back(gt_boxes_data[i*gt_dim+5]);
    gt_ignored.push_back(gt_boxes_data[i*gt_dim+6]);
  }
  
  // find the matched gt for each roi bb
  vector<int> max_gt_inds(num_rois); vector<float> max_overlaps(num_rois);
  for (int i = 0; i < num_rois; i++) {
    float maxop = -FLT_MAX; int maxid; bool exist_gt = false;
    for (int j = 0; j < num_gt_boxes; j++) {
      if (gt_boxes[j][0] != rois_boxes[i][0]) continue;
      exist_gt = true;
      float op = BoxIOU(rois_boxes[i][1], rois_boxes[i][2], rois_boxes[i][3], rois_boxes[i][4],
                 gt_boxes[j][1], gt_boxes[j][2], gt_boxes[j][3], gt_boxes[j][4], "IOU"); 
      if (op > maxop) {
        maxop = op; maxid = j;
      }
    }
    if (exist_gt) {
      max_gt_inds[i] = maxid; max_overlaps[i] = maxop;
    } else {
      max_gt_inds[i] = -1; max_overlaps[i] = 0;
    }
  }
  
  //select foreground rois with overlap >= fg_thr && is_ignored == false
  //select background rois with overlap within [fg_thr_lw,fg_thr_hg] && is_ignored == false
  vector<std::pair<int,int> > fg_inds_gtids, bg_inds_gtids, discard_bg_inds_gtids, keep_inds_gtids;
  for (int i = 0; i < num_rois; i++) {
    if (max_overlaps[i] >= fg_thr) {
      CHECK_GT(gt_labels[max_gt_inds[i]],0); //check if fg?
      if (gt_ignored[max_gt_inds[i]]) continue; //ignored
      fg_inds_gtids.push_back(std::make_pair(i,max_gt_inds[i]));
    } else if (max_overlaps[i]>=bg_thr_lw && max_overlaps[i]<bg_thr_hg) {
      bg_inds_gtids.push_back(std::make_pair(i,max_gt_inds[i]));
    } else {
      discard_bg_inds_gtids.push_back(std::make_pair(i,max_gt_inds[i]));
    }
  }
  int fg_rois_this_batch = std::min(fg_rois_per_batch,int(fg_inds_gtids.size()));
  if (fg_inds_gtids.size() > fg_rois_this_batch) {
    //random sampling
    caffe::rng_t* shuffle_rng = static_cast<caffe::rng_t*>(shuffle_rng_->generator());
    shuffle(fg_inds_gtids.begin(), fg_inds_gtids.end(), shuffle_rng);
    fg_inds_gtids.resize(fg_rois_this_batch);
  }
  int bg_rois_this_batch = batch_size_-fg_rois_this_batch;
  bg_rois_this_batch = std::min(bg_rois_this_batch,int(bg_inds_gtids.size()));
  if (bg_inds_gtids.size() > (batch_size_-fg_rois_this_batch)) {
    //random sampling
    caffe::rng_t* shuffle_rng = static_cast<caffe::rng_t*>(shuffle_rng_->generator());
    shuffle(bg_inds_gtids.begin(), bg_inds_gtids.end(), shuffle_rng);
    bg_inds_gtids.resize(bg_rois_this_batch);
  } else if (discard_bg_inds_gtids.size() > 0) {
    //pick up some samples from discarded pool
    int num_refind_bg_rois = batch_size_-fg_rois_this_batch-bg_inds_gtids.size();
    num_refind_bg_rois = std::min(num_refind_bg_rois,int(discard_bg_inds_gtids.size()));
    for (int i = 0; i < num_refind_bg_rois; i++) {
      bg_inds_gtids.push_back(discard_bg_inds_gtids[i]);
      bg_rois_this_batch++;
    }
  }
  
  int num_keep_rois = fg_rois_this_batch+bg_rois_this_batch;
  if (num_keep_rois < batch_size_) {
    int num_backup = batch_size_-num_keep_rois;
    LOG(INFO) << "sampled rois: " << num_keep_rois << ", random rois: "<<num_backup;
    //collect num_backup random bg boxes
    vector<vector<Dtype> > backup_boxes;
    while (backup_boxes.size() <= num_backup) {
      int img_id = caffe_rng_rand() % num_img_per_batch;
      int bb_x = caffe_rng_rand() % (img_width-32);
      int bb_y = caffe_rng_rand() % (img_height-32);
      int bb_width = caffe_rng_rand() % (img_width-bb_x);
      int bb_height = caffe_rng_rand() % (img_height-bb_y);
      bb_width = std::max(bb_width,32); bb_height = std::max(bb_height,32);

      vector<Dtype> bb(5);
      bb[0] = img_id; bb[1] = bb_x; bb[2] = bb_y; bb[3] = bb_width; bb[4] = bb_height;

      float maxop = -FLT_MAX;
      for (int j = 0; j < num_gt_boxes; j++) {
        // Skip any ground truth which is for a different image
        if (gt_boxes[j][0] != img_id) continue;

        float op = BoxIOU(bb[1], bb[2], bb[3], bb[4],
                        gt_boxes[j][1], gt_boxes[j][2], gt_boxes[j][3], gt_boxes[j][4], "IOU");
        if (op > maxop) maxop = op;
      }
      if (maxop >= fg_thr) continue;

      backup_boxes.push_back(bb);
    }
    for (int i = 0; i < num_backup; i++) {
      rois_boxes.push_back(backup_boxes[i]);
      int tmp_roi_id = rois_boxes.size()-1;
      bg_inds_gtids.push_back(std::make_pair(tmp_roi_id,-1));
      bg_rois_this_batch++;
    }
    num_keep_rois += num_backup;
  }
  CHECK_EQ(num_keep_rois,batch_size_);
  
  //append index and labels
  vector<Dtype> labels; 
  for (int i = 0; i < fg_rois_this_batch; i++) {
    keep_inds_gtids.push_back(fg_inds_gtids[i]);
    int tmplabel = gt_labels[fg_inds_gtids[i].second];
    CHECK_GT(tmplabel,0); labels.push_back(tmplabel);
  }
  for (int i = 0; i < bg_rois_this_batch; i++) {
    keep_inds_gtids.push_back(bg_inds_gtids[i]);
    labels.push_back(0);
  }
  
  //get the box regression target
  vector<vector<Dtype> > bbox_targets, size_targets; vector<vector<Dtype> > match_gt_boxes;
  for (int i = 0; i < num_keep_rois; i++) {
    Dtype bb_width, bb_height, bb_ctr_x, bb_ctr_y;
    Dtype gt_width, gt_height, gt_ctr_x, gt_ctr_y;
    Dtype targets_dx, targets_dy, targets_dw, targets_dh;
    vector<Dtype> bb_target(4); 
    vector<Dtype> match_gt_box(6);

    vector<Dtype> size_target(3);
    FrameObject current_frame_object;

    int bbid = keep_inds_gtids[i].first, gtid = keep_inds_gtids[i].second;  
    if (gtid >= 0) {

      bb_width = rois_boxes[bbid][3]; bb_height = rois_boxes[bbid][4];
      bb_ctr_x = rois_boxes[bbid][1]+0.5*bb_width;
      bb_ctr_y = rois_boxes[bbid][2]+0.5*bb_height;
      gt_width = gt_boxes[gtid][3]; gt_height = gt_boxes[gtid][4];
      gt_ctr_x = gt_boxes[gtid][1]+0.5*gt_width;
      gt_ctr_y = gt_boxes[gtid][2]+0.5*gt_height;
      //
      // If the ground truth width and height is equal to -1, this means
      // that ground truth is not available for this specific example
      // therefore skip it
      if (gt_width != -1 && gt_height != -1) {
        targets_dx = (gt_ctr_x - bb_ctr_x) / bb_width;
        targets_dy = (gt_ctr_y - bb_ctr_y) / bb_height;
        targets_dw = log(gt_width / bb_width);
        targets_dh = log(gt_height / bb_height);
        bb_target[0]=targets_dx; bb_target[1]=targets_dy;
        bb_target[2]=targets_dw; bb_target[3]=targets_dh;

        // bbox normalization
        if (do_bbox_norm) {
          bb_target[0] -= bbox_means[0]; bb_target[1] -= bbox_means[1];
          bb_target[2] -= bbox_means[2]; bb_target[3] -= bbox_means[3];
          bb_target[0] /= bbox_stds[0]; bb_target[1] /= bbox_stds[1];
          bb_target[2] /= bbox_stds[2]; bb_target[3] /= bbox_stds[3];
        }
      } else {
        bb_target[0] = -1;
        bb_target[1] = -1;
        bb_target[2] = -1;
        bb_target[3] = -1;
      }

      //positives for bbox evaluation
      if (labels[i] > 0) { 
        match_gt_box[0] = labels[i];
        match_gt_box[1] = gt_boxes[gtid][1]; match_gt_box[3] = gt_boxes[gtid][1]+gt_width-1; 
        match_gt_box[2] = gt_boxes[gtid][2]; match_gt_box[4] = gt_boxes[gtid][2]+gt_height-1; 
        CHECK_LT(bbid,int(max_overlaps.size())); CHECK_GE(max_overlaps[bbid],fg_thr); 
        match_gt_box[5] = max_overlaps[bbid];    
      }

      Eigen::Matrix3f P = proj_mat[rois_boxes[bbid][0]];
      Eigen::Matrix4f M = model_mat[rois_boxes[bbid][0]];

      // 3d bounding box
      if (output_3d_boxes_) {
        set_frame_object(current_frame, P, M, bbox_means, bbox_stds, gtid, bb_ctr_x, bb_ctr_y, bb_width, bb_height, current_frame_object);

        Eigen::Vector3f target_scale = gt_size[gtid];

        float mean_scale[] = { 3.88395449168, 1.52608343191, 1.62858986849 };
        float std_scale[] = { 0.42591674112, 0.136694962985, 0.102162061207 };

        size_target[0] = ((target_scale[0]) - mean_scale[0])/std_scale[0];
        size_target[1] = ((target_scale[1]) - mean_scale[1])/std_scale[1];
        size_target[2] = ((target_scale[2]) - mean_scale[2])/std_scale[2];
      }
    }
    bbox_targets.push_back(bb_target);
    size_targets.push_back(size_target);
    append_frame_object(current_frame, current_frame_object);
    match_gt_boxes.push_back(match_gt_box);
  }

  //prepare the outputs
  // rois
  top[sampled_rois_idx_]->Reshape(num_keep_rois, 5, 1, 1);
  Dtype* rois_data = top[sampled_rois_idx_]->mutable_cpu_data();
  // labels
  top[labels_idx_]->Reshape(num_keep_rois, 1, 1, 1);
  Dtype* labels_data = top[labels_idx_]->mutable_cpu_data();
  // targets
  WeightedFeatureData targets_data = get_weighted_feature_top_data(top, bbox_targets_idx_, num_keep_rois, 4*cls_num_);

  WeightedFeatureData size_data;
  FrameData current_frame_data;

  if (output_3d_boxes_) {

    size_data = get_weighted_feature_top_data(top, size_targets_idx_, num_keep_rois, 3*cls_num_);
    current_frame_data = get_frame_top_data(current_frame_info, top, num_keep_rois);
  }
  // matched gt boxes
  top[matched_gt_boxes_idx_]->Reshape(num_keep_rois, 6, 1, 1);
  Dtype* match_gt_boxes_data = top[matched_gt_boxes_idx_]->mutable_cpu_data();
  caffe_set(top[matched_gt_boxes_idx_]->count(), Dtype(0), match_gt_boxes_data);
  // sample weights for softmax loss
  if (has_sample_weight_) {
    top[sample_weights_idx_]->Reshape(num_keep_rois, 1, 1, 1);
    Dtype* sample_weights_data = top[sample_weights_idx_]->mutable_cpu_data();
    caffe_set(top[sample_weights_idx_]->count(), Dtype(1), sample_weights_data);
    Dtype pos_weight_sum = Dtype(0), neg_weight_sum = Dtype(0); 
    if (iou_weighted) {
      for (int i = 0; i < num_keep_rois; i++) {
        sample_weights_data[i] = labels[i]>0? match_gt_boxes[i][5]:1;
      }
    }
    for (int i = 0; i < num_keep_rois; i++) {
      if (labels[i]>0) pos_weight_sum += sample_weights_data[i]; 
      else neg_weight_sum += sample_weights_data[i]; 
    }
    for (int i = 0; i < num_keep_rois; i++) {
      if (labels[i] > 0) {
        if (pos_weight_sum != 0) 
          sample_weights_data[i] *= (fg_fraction*num_keep_rois/pos_weight_sum); 
      } else {
        if (pos_weight_sum != 0) 
          sample_weights_data[i] *= ((1-fg_fraction)*num_keep_rois/neg_weight_sum); 
      }
    }    
  }

  Dtype *roi_height_data;
  top[roi_height_idx_]->Reshape({num_keep_rois, cls_num_});
  roi_height_data = top[roi_height_idx_]->mutable_cpu_data();
  caffe_set(top[roi_height_idx_]->count(), Dtype(0), roi_height_data);
  
  for (int i = 0; i < num_keep_rois; i++) {
    int cls_id = labels[i], rois_id = keep_inds_gtids[i].first;
    labels_data[i] = cls_id;
    //rois = (img_id, x1, y1, x2, y2)
    const int drois_dim = 5;
    rois_data[i*drois_dim] = rois_boxes[rois_id][0];
    rois_data[i*drois_dim+1] = rois_boxes[rois_id][1];
    rois_data[i*drois_dim+2] = rois_boxes[rois_id][2];
    rois_data[i*drois_dim+3] = rois_boxes[rois_id][1]+rois_boxes[rois_id][3]-1;
    rois_data[i*drois_dim+4] = rois_boxes[rois_id][2]+rois_boxes[rois_id][4]-1;

    if (output_3d_boxes_)
      current_frame_data.quadrant[i] = cls_id == 0 ? -1 /*background*/ : current_frame.quadrant_targets[i][0];

    if (cls_id == 0) continue;

    fill_weighted_data(targets_data, bbox_targets[i], 4*(i*cls_num_+cls_id), 4);
    if (output_3d_boxes_) {
      fill_weighted_data(size_data, size_targets[i], 3*(i*cls_num_+cls_id), 3);
      fill_weighted_data(current_frame_data.location, current_frame.location_targets[i], 2*(i*cls_num_+cls_id), 2);
      fill_weighted_data(current_frame_data.distance, current_frame.distance_targets[i], (i*cls_num_+cls_id), 1);
      fill_weighted_data(current_frame_data.orientation, current_frame.orientation_targets[i], 2*(i*cls_num_+cls_id), 2);
    }

    roi_height_data[i*cls_num_+cls_id] = rois_boxes[rois_id][4];

    for (int j = 0; j < 6; j++) {
      // (label, x1, y1, x2, y2, overlap)
      match_gt_boxes_data[i*6+j] = match_gt_boxes[i][j];
    }
  }
}

INSTANTIATE_CLASS(ProposalTargetLayer);
REGISTER_LAYER_CLASS(ProposalTarget);

}  // namespace caffe
