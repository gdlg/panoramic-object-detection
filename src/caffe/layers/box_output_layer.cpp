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

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/box_output_layer.hpp"

namespace caffe {
    
template <typename Dtype>
void BoxOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BoxOutputParameter box_output_param = this->layer_param_.box_output_param();
  fg_thr_ = box_output_param.fg_thr();
  iou_thr_ = box_output_param.iou_thr();
  nms_type_ = box_output_param.nms_type();
  num_param_set_ = box_output_param.num_param_set();
  output_proposal_with_score_ = (top.size() == 2);
  ringpad_ = box_output_param.ringpad();
}

template <typename Dtype>
void BoxOutputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //dummy reshape
  top[0]->Reshape(1, 4*num_param_set_+1, 1, 1);
  if (output_proposal_with_score_) {
    top[1]->Reshape(1, 4*num_param_set_+2, 1, 1);
  }
}

template <typename Dtype>
void nmsMax(vector<vector<Dtype> >& bbs, const float overlap,
        const string& mode) {
  //bbs[i] = [batch_idx x y w h sc];
  // for each i suppress all j st j>i and area-overlap>overlap

  auto keep_end = begin(bbs);
  auto bbs_end = end(bbs);
  for (auto iter = begin(bbs); iter != bbs_end; ++iter) {
    bool keep = true;
    auto& bbs1 = *iter;
    for (auto iter2 = begin(bbs); iter2 != keep_end; ++iter2) {
      auto& bbs2 = *iter2;
      Dtype o = BoxIOU(bbs1[1], bbs1[2], bbs1[3], bbs1[4],
                 bbs2[1], bbs2[2], bbs2[3], bbs2[4], mode);
      keep = o <= overlap;
      if (!keep)
          break;
    }

    if (keep) {
        if (keep && keep_end != iter) {
            *keep_end = std::move(*iter);
        }
        ++keep_end;
    }
  }

  bbs.erase(keep_end, end(bbs));
}

unsigned round_up_to_pow2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

template <typename Dtype>
void BoxOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  thread_local vector<vector<Dtype> > batch_boxes;
  batch_boxes.clear();

  int num_batch_boxes = 0;
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int bottom_num = bottom.size();
  const int cls_num = channels - num_param_set_ * 4;
  float field_whr = this->layer_param_.box_output_param().field_whr();
  float field_xyr = this->layer_param_.box_output_param().field_xyr();
  const Dtype min_whr = log(Dtype(1)/field_whr), max_whr = log(Dtype(field_whr));
  const Dtype min_xyr = Dtype(-1)/field_xyr, max_xyr = Dtype(1)/field_xyr;
  const int max_nms_num = this->layer_param_.box_output_param().max_nms_num(); 
  const int max_post_nms_num = this->layer_param_.box_output_param().max_post_nms_num(); 
  const float min_size = this->layer_param_.box_output_param().min_size();
  
  CHECK_EQ(bottom_num, this->layer_param_.box_output_param().field_h_size());
  CHECK_EQ(bottom_num, this->layer_param_.box_output_param().field_w_size());
  CHECK_EQ(bottom_num, this->layer_param_.box_output_param().downsample_rate_size());
  vector<float> field_ws, field_hs, downsample_rates;
  for (int i = 0; i < bottom_num; i++) {
    field_ws.push_back(this->layer_param_.box_output_param().field_w(i));
    field_hs.push_back(this->layer_param_.box_output_param().field_h(i));
    downsample_rates.push_back(this->layer_param_.box_output_param().downsample_rate(i));
  }
  
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

  for (int i = 0; i < num; i++) {
    vector<vector<Dtype> > boxes;
    std::vector<std::pair<Dtype, int> > score_idx_vector;
    score_idx_vector.clear();
    int bb_count = 0;
    for (int j = 0; j < bottom_num; j++) {
      const Dtype* bottom_data = bottom[j]->cpu_data();
      int bottom_dim = bottom[j]->count() / num;
      int width = bottom[j]->width(), height = bottom[j]->height();
      int img_width = width*downsample_rates[j], img_height = height*downsample_rates[j];
      int spatial_dim = width*height;
      
      for (int id = 0; id < spatial_dim; id++) {
        const int base_idx = i*bottom_dim+id;
        const int coord_idx = base_idx+cls_num*spatial_dim;
        const int h = id / width, w = id % width; 
        Dtype fg_score = -FLT_MAX;
        // get the max score across positive classes
        for (int k = 1; k < cls_num; k++) {
          fg_score = std::max(fg_score,bottom_data[base_idx+k*spatial_dim]);
        }
        fg_score -= bottom_data[base_idx]; // max positive score minus negative score
      
        if (fg_score >= fg_thr_) {
          vector<Dtype> bb(2+num_param_set_*4);

          bb[0] = i;

          for (int k=0; k < num_param_set_; ++k) {
            Dtype bbx, bby, bbw, bbh;
            bbx = bottom_data[coord_idx+(4*k+0)*spatial_dim];
            bby = bottom_data[coord_idx+(4*k+1)*spatial_dim];
            bbw = bottom_data[coord_idx+(4*k+2)*spatial_dim];
            bbh = bottom_data[coord_idx+(4*k+3)*spatial_dim];

            // bbox de-normalization
            if (do_bbox_norm) {
              bbx *= bbox_stds[0]; bby *= bbox_stds[1];
              bbw *= bbox_stds[2]; bbh *= bbox_stds[3];
              bbx += bbox_means[0]; bby += bbox_means[1];
              bbw += bbox_means[2]; bbh += bbox_means[3];
            }

            bbx = std::max(min_xyr,bbx); bbx = std::min(max_xyr,bbx);
            bby = std::max(min_xyr,bby); bby = std::min(max_xyr,bby);
            bbx = bbx*field_ws[j] + (w+Dtype(0.5))*downsample_rates[j];
            bby = bby*field_hs[j] + (h+Dtype(0.5))*downsample_rates[j];
        
            bbw = std::max(min_whr,bbw); bbw = std::min(max_whr,bbw);
            bbh = std::max(min_whr,bbh); bbh = std::min(max_whr,bbh);
            bbw = field_ws[j] * exp(bbw); bbh = field_hs[j] * exp(bbh);

            // Snap the bbw and bbh to 7 times a multiple of two (round up)
            //const int size_alignment = 28;
            //bbw = (((int)bbw + size_alignment - 1) / size_alignment) * size_alignment;
            //bbh = (((int)bbh + size_alignment - 1) / size_alignment) * size_alignment;

            const int size_alignment = 7;
            int bbw_factor = (bbw + size_alignment - 1) / size_alignment;
            bbw = round_up_to_pow2(bbw_factor) * size_alignment;
            int bbh_factor = (bbh + size_alignment - 1) / size_alignment;
            bbh = round_up_to_pow2(bbh_factor) * size_alignment;

            bbx = bbx - bbw/Dtype(2); bby = bby - bbh/Dtype(2);
            if (!ringpad_) {
              bbx = std::max(bbx,Dtype(0));
            }
            bby = std::max(bby,Dtype(0));

            // Snap bbx  and bby (round down)
            const int pos_alignment = 4;
            bbx = ((int)bbx / pos_alignment) * pos_alignment;
            bby = ((int)bby / pos_alignment) * pos_alignment;

            if (!ringpad_) {
              bbw = std::min(bbw,img_width-bbx);
            }
            bbh = std::min(bbh,img_height-bby);
            bb[4*k+1] = bbx; bb[4*k+2] = bby; bb[4*k+3] = bbw; bb[4*k+4] = bbh;
          }

          bb[4*num_param_set_+1] = fg_score;
          if (bb[3] >= min_size && bb[4] >= min_size) {
            boxes.push_back(bb);
            score_idx_vector.push_back(std::make_pair(fg_score, bb_count++));
          }
        }
      }
    }
    
    DLOG(INFO) << "The number of boxes before NMS: " << boxes.size();
    if (boxes.size()<=0) continue;
    //ranking decreasingly
    std::sort(score_idx_vector.begin(),score_idx_vector.end(),std::greater<std::pair<Dtype, int> >());
    //
    //keep top N boxes before NMS
    if (max_nms_num > 0 && bb_count > max_nms_num) {
      score_idx_vector.resize(max_nms_num);
    }

    thread_local vector<vector<Dtype> > new_boxes;
    new_boxes.clear();

    for (int kk = 0; kk < score_idx_vector.size(); kk++) {
      new_boxes.push_back(std::move(boxes[score_idx_vector[kk].second]));
    }
    std::swap(boxes, new_boxes);

    //NMS
    nmsMax(boxes, iou_thr_, nms_type_);
    int num_new_boxes =  boxes.size();
    if (max_post_nms_num > 0 && num_new_boxes > max_post_nms_num) {
      num_new_boxes = max_post_nms_num;
    }
    for (int kk = 0; kk < num_new_boxes; kk++) {
      batch_boxes.push_back(std::move(boxes[kk]));
    }
    num_batch_boxes += num_new_boxes;
  }
  
  CHECK_EQ(num_batch_boxes,batch_boxes.size());
  // output rois [batch_idx x1 y1 x2 y2] for each roi_pooling layer
  if (num_batch_boxes <= 0) {
    // for special case when there is no box
    top[0]->Reshape(1, 4*num_param_set_+1, 1, 1);
    Dtype* top_boxes = top[0]->mutable_cpu_data();
    top_boxes[0]=0;
    for (size_t j = 0; j < num_param_set_; j++) {
        top_boxes[4*j+1]=1;
        top_boxes[4*j+2]=1;
        top_boxes[4*j+3]=10;
        top_boxes[4*j+4]=10;
    }
  } else {
    const int n = 1+num_param_set_*4;
    top[0]->Reshape(num_batch_boxes, n, 1, 1);
    Dtype* top_boxes = top[0]->mutable_cpu_data();
    for (int i = 0; i < num_batch_boxes; i++) {
      CHECK_EQ(batch_boxes[i].size(),2+num_param_set_*4);
      top_boxes[i*n] = batch_boxes[i][0];
      for (int k = 0; k < num_param_set_; k++)
      {
        top_boxes[i*n+4*k+1] = batch_boxes[i][4*k+1];
        top_boxes[i*n+4*k+2] = batch_boxes[i][4*k+2];
        top_boxes[i*n+4*k+3] = batch_boxes[i][4*k+1]+batch_boxes[i][4*k+3];
        top_boxes[i*n+4*k+4] = batch_boxes[i][4*k+2]+batch_boxes[i][4*k+4];
      }
    }
  }
  // output proposals+scores [batch_idx x1 y1 x2 y2 px1 py1 px2 py2 score] for proposal detection
  if (output_proposal_with_score_) {
    const int n = 2+num_param_set_*4;
    if (num_batch_boxes <= 0) {
      // for special case when there is no box
      top[1]->Reshape(1, n, 1, 1);
      Dtype* top_boxes_scores = top[1]->mutable_cpu_data();
      caffe_set(top[1]->count(), Dtype(0), top_boxes_scores); 
    } else {
      top[1]->Reshape(num_batch_boxes, n, 1, 1);
      Dtype* top_boxes_scores = top[1]->mutable_cpu_data();
      for (int i = 0; i < num_batch_boxes; i++) {
        CHECK_EQ(batch_boxes[i].size(),n);
        top_boxes_scores[i*n] = batch_boxes[i][0];
        for (int k = 0; k < num_param_set_; ++k) {
            top_boxes_scores[i*n+4*k+1] = batch_boxes[i][4*k+1];
            top_boxes_scores[i*n+4*k+2] = batch_boxes[i][4*k+2];
            top_boxes_scores[i*n+4*k+3] = batch_boxes[i][4*k+1]+batch_boxes[i][4*k+3];
            top_boxes_scores[i*n+4*k+4] = batch_boxes[i][4*k+2]+batch_boxes[i][4*k+4];
        }
        top_boxes_scores[i*n+n-1] = batch_boxes[i][n-1];
      }
    }
  }
}

INSTANTIATE_CLASS(BoxOutputLayer);
REGISTER_LAYER_CLASS(BoxOutput);

}  // namespace caffe
