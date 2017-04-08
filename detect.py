import os
import sys
import time
import random
import math
import argparse

this_dir = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(this_dir, 'python'))

import caffe
import scipy
import numpy as np
import cv2
from nms.cpu_nms import cpu_nms
from nms.gpu_nms import gpu_nms

mode = "car"

crop_pano = True


if mode == "all":
    cls_ids = [1,2,3,5,6]
    cls = ['bg','Car','Van','Truck','Tram', 'Pedestrian', 'Cyclist']
elif mode == "car":
    cls_ids = [1,2,3,4]
    cls = ['bg','Car','Van','Truck','Tram']
elif mode == "ped_cyc":
    cls_ids = [1,2]
    cls = ['bg', 'Pedestrian', 'Cyclist']

num_cls = len(cls)

mu = np.array([104,117,123])

proposals_thresh = -10

bbox_means = [0, 0, 0, 0];
bbox_stds = [0.1, 0.1, 0.2, 0.2];

stat_mean_xy = [0, 0]
stat_var_xy = [0.1, 0.1]

stat_mean_scale = [3.88395449168, 1.52608343191, 1.62858986849]
stat_var_scale = [0.42591674112, 0.136694962985, 0.102162061207]

nms_thresh = 0.5

def spherical_project(P,x,y,z):
    lon = np.arctan2(x,z)
    lat = np.arcsin(y/np.sqrt(x**2+y**2+z**2))

    uv = np.matmul(P,np.stack([lon,lat,np.ones_like(lon)]))
    return (uv[0],uv[1])


def wrapped_line(image, p1, p2, colour, thickness):
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    _p1 = np.array(p1)
    _p2 = np.array(p2)

    dist1 = np.linalg.norm(_p1 - _p2)

    p1b = np.array([p1[0]+image.shape[1], p1[1]])
    p2b = np.array([p2[0]-image.shape[1], p2[1]])

    dist2 = np.linalg.norm(_p1 - p2b)

    if dist1 < dist2:
        cv2.line(image, p1, p2, colour, thickness)
    else:
        cv2.line(image, p1, tuple(p2b), colour, thickness)
        cv2.line(image, tuple(p1b), p2, colour, thickness)

def draw_spherical_project_line(P):
    def body(image, p1, p2, colour, thickness):
        quality=10
        t = np.linspace(0,1,num=quality)

        p1 = np.expand_dims(p1,1)
        p2 = np.expand_dims(p2,1)

        points = p1 * t + (1-t) * p2
        uv_x, uv_y = spherical_project(P,points[0], points[1], points[2])
        
        uv_x = uv_x.astype(np.int32)
        uv_y = uv_y.astype(np.int32)

        for t in range(quality-1):
            wrapped_line(image, (uv_x[t],uv_y[t]), (uv_x[t+1], uv_y[t+1]), colour, thickness)
    return body


def draw_cube(result_image, P, pred_xyz, pred_ry, pred_size, colour = (0,255,0), draw_spherical_lines=False):
    rotation_matrix = np.array([[np.cos(pred_ry), 0, np.sin(pred_ry), pred_xyz[0]], [0, 1, 0, pred_xyz[1]], [-np.sin(pred_ry), 0, np.cos(pred_ry), pred_xyz[2]], [0, 0, 0, 1]])
    if draw_spherical_lines:
        bbox_proj_mat = rotation_matrix
        vertices = np.empty([2,2,2,3])
        lineFunc = draw_spherical_project_line(P)
    else:
        bbox_proj_mat = np.matmul(P, rotation_matrix);
        vertices = np.empty([2,2,2,2])
        lineFunc = cv2.line
    
    draw_centre = False
    if draw_centre:
        if draw_spherical_lines:
            centre_project = np.array(spherical_project(P, pred_xyz[0], pred_xyz[1], pred_xyz[2]))
        else:
            centre_project = np.matmul(P, np.concatenate((pred_xyz, [1])))
            centre_project[0] /= centre_project[2]
            centre_project[1] /= centre_project[2]
            centre_project = centre_project[:2]
        cv2.circle(result_image, tuple(centre_project.astype(np.int32)), 3, (0,255,0))

    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                v = np.array([(k-0.5)*pred_size[0], (-l)*pred_size[1], (m-0.5)*pred_size[2], 1.])
                v = np.matmul(bbox_proj_mat, v)

                if draw_spherical_lines:
                    vertices[k,l,m,:] = v[:3]
                else:
                    vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    if not draw_spherical_lines:
        vertices = vertices.astype(np.int32)

    for k in [0, 1]:
        for l in [0, 1]:
            for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                lineFunc(result_image, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=2)
    for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
        lineFunc(result_image, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=2)

def initialization(model, weights):
    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(model,
                    weights,
                    caffe.TEST)
    return net

class ObjFrame:
    def __init__(self, data, idx):
        self.bbox_pred = data.bbox_pred[idx]
        self.pred_xyz = data.pred_xyz[idx]
        self.pred_ry = data.pred_ry[idx]
        self.pred_size = data.pred_size[idx]
        self.tight_proposal = data.tight_proposal[idx]

    def draw_cube(self, result_image, P, colour, draw_spherical_project_line):
        draw_cube(result_image, P, self.pred_xyz, self.pred_ry, self.pred_size, colour, draw_spherical_project_line)

    def draw_proposal(self, result_image, colour):
        cv2.rectangle(result_image, (int(self.tight_proposal[0]), int(self.tight_proposal[1])), (int(self.tight_proposal[2]), int(self.tight_proposal[3])), colour, 2)

    def draw_bbox(self, result_image, colour):
        bb = self.bbox_pred
        cv2.rectangle(result_image, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), colour, 2)

class Object:
    def __init__(self, cur, score, idx, cls_id):
        self.score = score
        self.cur = ObjFrame(cur, idx)
        self.cls_id = cls_id

    def generate_kitti_bench_detection(self, out):
        pred_ry = self.cur.pred_ry
        bbox_pred = self.cur.bbox_pred
        pred_size = self.cur.pred_size
        pred_xyz = self.cur.pred_xyz
        score = self.score * 1000

        pred_alpha = pred_ry - math.atan2(pred_xyz[0], pred_xyz[2]);
        objtype = cls[self.cls_id]

        out.write("%s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f\n" %
                (objtype, -1, -1, pred_alpha, bbox_pred[0], bbox_pred[1], bbox_pred[0]+bbox_pred[2], bbox_pred[1]+bbox_pred[3], pred_size[1], pred_size[2], pred_size[0], pred_xyz[0], pred_xyz[1], pred_xyz[2], pred_ry, score))

class Frame:
    def __init__(self, rectilinear_mode, image_size, ratios, P, M, tight_proposal, proposal_ctr_x, proposal_ctr_y, proposal_width, proposal_height, cls_id, bbox_preds, location_preds, orientation_preds, quadrant_preds, distance_pred_a, distance_pred_b):

        focal_length = abs(P[0,0])
        epipole_y = abs(P[1,2])

        bbox_pred = bbox_preds[:,:4]
        orientation_pred = orientation_preds[:,:]
        location_pred = location_preds[:,:2]

        bbox_pred = bbox_pred * bbox_stds + bbox_means

        tx = bbox_pred[:,0] * proposal_width + proposal_ctr_x
        ty = bbox_pred[:,1] * proposal_height + proposal_ctr_y
        tw = proposal_width * np.exp(bbox_pred[:,2]);
        th = proposal_height * np.exp(bbox_pred[:,3]);
        tx = tx - tw/2.
        ty = ty - th/2.

        tx /= ratios[0]
        ty /= ratios[1]
        tw /= ratios[0]
        th /= ratios[1]

        tx2 = tx + tw
        ty2 = ty + th

        tx = np.clip(tx, 0, image_size[0])
        ty = np.clip(ty, 0, image_size[1])
        tx2 = np.clip(tx2, 0, image_size[0])
        ty2 = np.clip(ty2, 0, image_size[1])
        tw = tx2 - tx
        th = ty2 - ty

        # Centre computation
        targets_xy = location_pred[:,:2];
        targets_xy = targets_xy * stat_var_xy + stat_mean_xy
        pred_ctr_xy = targets_xy * np.column_stack((proposal_width, proposal_height)) + np.column_stack((proposal_ctr_x, proposal_ctr_y));


        targets_z = focal_length / (distance_pred_a[:] * proposal_height + distance_pred_b[:]);

        Pinv = np.linalg.inv(P)
        Minv = np.linalg.inv(M)

        if rectilinear_mode:
            hxyz = np.concatenate((pred_ctr_xy, np.ones_like(targets_z)[:,np.newaxis], np.ones_like(targets_z)[:,np.newaxis]), 1)

            pred_xyz = np.transpose(np.matmul(Pinv, np.transpose(hxyz)));
            pred_xyz = pred_xyz[:,:3]

            pred_xyz /= np.linalg.norm(pred_xyz, ord=2, axis=1)[:,np.newaxis]
            pred_xyz *= targets_z[:,np.newaxis]

            pred_xyz = np.concatenate((pred_xyz, np.ones_like(targets_z)[:, np.newaxis]), 1)
            pred_xyz = np.transpose(np.matmul(Minv, np.transpose(pred_xyz)));
        else:
            ctr_xy = np.concatenate((pred_ctr_xy, np.ones_like(targets_z)[:,np.newaxis]), 1)
            ctr_xy = np.transpose(np.matmul(Pinv, np.transpose(ctr_xy)));
            lon = ctr_xy[:,0]
            lat = ctr_xy[:,1]

            targets_r = targets_z

            targets_y = targets_r * np.sin(lat)

            targets_r1 = np.sqrt(targets_r ** 2 - targets_y ** 2)

            targets_x = targets_r1 * np.sin(lon)
            targets_z = targets_r1 * np.cos(lon)

            pred_xyz = np.transpose(np.stack((targets_x, targets_y, targets_z, np.ones_like(targets_z))))
            pred_xyz = np.transpose(np.matmul(Minv, np.transpose(pred_xyz)));

        # Orientation computation

        quadrant_preds_exp = np.exp(quadrant_preds);
        quadrant_preds = quadrant_preds_exp / np.sum(quadrant_preds_exp,1)[:,np.newaxis];
        quadrant_index = np.argmax(quadrant_preds, 1)


        angle_cos = np.sqrt(orientation_pred[:,0]);
        angle_sin = np.sqrt(orientation_pred[:,1]);
        sign_table = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        signs = sign_table[quadrant_index,:];

        pred_alpha = np.arctan2(signs[:,1] * angle_sin, signs[:,0] * angle_cos)
        pred_ry = pred_alpha + np.arctan2(pred_xyz[:,0], pred_xyz[:,2]);

        self.bbox_pred = np.stack((tx,ty,tw,th),-1)
        self.targets_z = targets_z
        self.pred_xyz = pred_xyz
        self.pred_alpha = pred_alpha
        self.pred_ry = pred_ry

        self.tight_proposal = tight_proposal.copy()
        self.tight_proposal[:,(0,2)] /= ratios[0]
        self.tight_proposal[:,(1,3)] /= ratios[1]


class Data:
    pass


def im_detect(net, P, M, rectilinear_mode, cur_im):

    if rectilinear_mode:
        net_image_size = (1920,576)
    else:
        if crop_pano:
            net_image_size = (3502,636)
        else:
            net_image_size = (3702,1536)

    Pnet = np.copy(P)

    image_size = (cur_im.shape[1], cur_im.shape[0])
    ratios = (net_image_size[0]/float(image_size[0]), net_image_size[1]/float(image_size[1]))
    Pnet[0,:] *= ratios[0]
    Pnet[1,:] *= ratios[1]

    show_image = np.copy(cur_im)
    cur_im = scipy.misc.imresize(cur_im, (net_image_size[1],net_image_size[0])).astype(np.float32)

    # Concatenate the two images together
    input_pair = cur_im

    # Normalize the image
    input_pair = input_pair - mu

    input_pair = np.transpose(input_pair, (2,0,1))
    input_pair = input_pair[np.newaxis]

    forward_kwargs = {'data': input_pair.astype(np.float32, copy=False)}
    outputs = net.forward(**forward_kwargs)

    proposals_pred = outputs['proposals_score'][:,1:,0,0]

    x1 = proposals_pred[:,0]
    y1 = proposals_pred[:,1]
    x2 = proposals_pred[:,2]
    y2 = proposals_pred[:,3]

    proposal_width = x2 - x1
    proposal_height = y2 - y1
    proposal_ctr_x = x1 + 0.5 * proposal_width
    proposal_ctr_y = y1 + 0.5 * proposal_height

    score = proposals_pred[:,-1]

    num_proposals = score.shape[0]

    cls_preds = outputs['cls_pred']

    exp_scores = np.exp(cls_preds)
    sum_exp_scores = np.sum(exp_scores,1)
    probs = exp_scores / sum_exp_scores[:,np.newaxis]

    # Skip the background when computing argmax
    det_most_likely_cls_id = np.argmax(probs[:,1:], 1) + 1

    keep_mask = np.logical_and(np.logical_and(score > proposals_thresh, np.not_equal(proposal_width,0)), np.not_equal(proposal_height,0))

    proposal_width = proposal_width[keep_mask]
    proposal_height = proposal_height[keep_mask]
    proposal_ctr_x = proposal_ctr_x[keep_mask]
    proposal_ctr_y = proposal_ctr_y[keep_mask]

    tight_proposal = proposals_pred[keep_mask,:]
    quadrant_preds = outputs['quadrant_pred'][keep_mask,:]

    select_best_class = False
    if select_best_class:
        det_cls_id = det_most_likely_cls_id[keep_mask]

        keep_mask = np.where(keep_mask)[0] * num_cls + det_cls_id
    else:
        keep_mask = np.where(keep_mask)[0] * num_cls
        det_cls_id = np.tile(cls_ids, keep_mask.size)
        keep_mask = np.repeat(keep_mask, len(cls_ids))

        keep_mask = keep_mask + det_cls_id

        proposal_ctr_x = np.repeat(proposal_ctr_x, len(cls_ids))
        proposal_ctr_y = np.repeat(proposal_ctr_y, len(cls_ids))
        proposal_width = np.repeat(proposal_width, len(cls_ids))
        proposal_height = np.repeat(proposal_height, len(cls_ids))
        quadrant_preds = np.repeat(quadrant_preds, len(cls_ids), axis=0)
        tight_proposal = np.repeat(tight_proposal, len(cls_ids), axis=0)

    probs = probs.reshape(num_proposals*num_cls)[keep_mask]

    size_preds = outputs['size_pred'].reshape(num_proposals*num_cls,-1)[keep_mask,:]
    bbox_preds = outputs['bbox_pred'].reshape(num_proposals*num_cls,-1)[keep_mask,:]
    location_preds = outputs['location_pred'].reshape(num_proposals*num_cls,-1)[keep_mask,:]
    orientation_preds = outputs['orientation_pred'].reshape(num_proposals*num_cls,-1)[keep_mask,:]
    distance_pred_a = outputs['distance_pred_a'].reshape(num_proposals*num_cls)[keep_mask]
    distance_pred_b = outputs['distance_pred_b'].reshape(num_proposals*num_cls)[keep_mask]

    all_objects = dict()

    targets_size = size_preds;
    targets_size = targets_size * stat_var_scale + stat_mean_scale
    pred_size = targets_size

    cur = Frame(rectilinear_mode, image_size, ratios, Pnet, M, tight_proposal[:,:4], proposal_ctr_x, proposal_ctr_y, proposal_width, proposal_height, det_cls_id, bbox_preds, location_preds, orientation_preds, quadrant_preds, distance_pred_a, distance_pred_b)
    cur.pred_size = pred_size

    cls_dets = np.hstack((cur.bbox_pred, probs[:,np.newaxis]))
    cls_dets[:,2] += cls_dets[:,0]
    cls_dets[:,3] += cls_dets[:,1]

    if cls_dets.shape[0] > 0:
        nms_keep = gpu_nms(cls_dets.astype(np.float32), nms_thresh)
    else:
        nms_keep = []

    all_objects = dict()
    for c in cls_ids:
        all_objects[c] = []

    for idx in nms_keep:
        if det_cls_id[idx] in cls_ids:
            all_objects[det_cls_id[idx]].append(Object(cur, probs[idx], idx, det_cls_id[idx]))


    return all_objects


def run_on_detection_dir(net, image_dir, output_dir, display_image=False, rectilinear_mode=False):

    image_list = sorted([os.path.splitext(img)[0] for img in os.listdir(image_dir)])
    try:
        os.mkdir(output_dir)
    except:
        pass

    results_image_dir = os.path.join(output_dir, "image_2")
    results_det_dir = os.path.join(output_dir, "data")
    try:
        os.mkdir(results_image_dir)
    except:
        pass
    try:
        os.mkdir(results_det_dir)
    except:
        pass

    for image_no in image_list:
        print(image_no)

        if rectilinear_mode:
            calib_dir = os.path.join(image_dir, "../calib")
            calib_path = os.path.join(calib_dir,"000000.txt")
            for line in open(calib_path).readlines():
                data = line.rstrip().split(' ')
                if data[0] == "P2:":
                    P_kitti = np.reshape(np.array([float(d) for d in data[1:]] + [0,0,0,1]), (4,4))
            P = P_kitti

            ## Separate the reference frame transformation
            ## from the actual projection.

            M = np.array(
                [[1, 0, 0, P_kitti[0,3]/P_kitti[0,0]],
                 [0, 1, 0, P_kitti[1,3]/P_kitti[1,1]],
                 [0, 0, 1, P_kitti[2,3]/P_kitti[2,2]],
                 [0, 0, 0, 1]])

            P[:3,3] = 0
        else:
            if large_pano:
                P = np.array(
                [[2048./(2.*math.pi), 0, 2048./2.],
                 [0, 1024./(math.pi), 1024./2.],
                 [0, 0, 1]])
            elif crop_pano:
                P = np.array(
                [[2048./(2.*math.pi), 0, 2048./2.],
                 [0, 1024./(math.pi), 88.],
                 [0, 0, 1]])
            else:
                P = np.array(
                [[469./(2.*0.720694854453), 0, 469./2.],
                 [0, 168./(2.*0.259220078424), 168./2.],
                 [0, 0, 1]])

            M = np.identity(4)



        image_fullpath = os.path.join(image_dir, image_no + ".jpg")
        if not os.path.exists(image_fullpath):
            image_fullpath = os.path.join(image_dir, image_no + ".png")
        cur_image = cv2.imread(image_fullpath)

        out_detections = open(os.path.join(results_det_dir, image_no + ".txt"), "w")

        result_image = np.copy(cur_image)

        all_objects = im_detect(net, P, M, rectilinear_mode, cur_image)


        for objects in all_objects.values():
            for obj in objects:
                obj.generate_kitti_bench_detection(out_detections)

        min_score = .5

        for cls_id,objects in all_objects.items():
            if cls[cls_id] not in ["Car", "Pedestrian", "Cyclist", "Van", "Truck"]:
                continue

            for obj in objects:
                if obj.score > min_score:
                    colour = (255,255,255)
                    #colour = (obj.score * colour[0], obj.score * colour[1], obj.score * colour[2])
                    obj.cur.draw_cube(result_image, P, colour, not rectilinear_mode)
                    #obj.cur.draw_bbox(result_image, colour)
                    #obj.cur.draw_proposal(result_image, colour)

        cv2.imwrite(os.path.join(results_image_dir, image_no + ".png"), result_image)

        if display_image:
            cv2.imshow("My image",result_image)
            key = cv2.waitKey(10)
            if key == 99: # 'q'
                sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Generate detections from a set of images.")
    parser.add_argument('--weights', required=True,
            help='The model weights (.caffemodel)')
    parser.add_argument('--input', required=True,
            help='Input image directory')
    parser.add_argument('--output', required=True,
            help='Output directory to write detections')

    default_model="examples/inference/deploy_crop.prototxt"
    parser.add_argument('--model', default=default_model,
            help='The model weights (.caffemodel), default: '+default_model)
    parser.add_argument('--display', action='store_true',
            help='Display output images')
    parser.add_argument('--rectilinear', action='store_true',
            help='Disable panoramic mode')
    args = parser.parse_args()

    net = initialization(os.path.expanduser(args.model), os.path.expanduser(args.weights))
    run_on_detection_dir(net, os.path.expanduser(args.input), os.path.expanduser(args.output), display_image=args.display, rectilinear_mode=args.rectilinear)

if __name__ == "__main__":
    main()

