import sys
import os
import math
import numpy as np
import itertools

import transformations

script_directory = os.path.dirname(os.path.realpath(sys.argv[0]))

def panoramic_projection_matrix():
    return np.array([
        [ 325.94932345,    0.,          230.45443645],
        [   0.,          325.94932345,   81.45081126],
        [   0.,            0.,            1.        ]])

def spherical_project(x,y,z):

    P=panoramic_projection_matrix()

    M = np.array([
        [1., 0., 0., 44.85728/721.5377],
        [0., 1., 0., 0.2163791/721.5377],
        [0., 0., 1., 0.002745884],
        [0., 0., 0., 1.]])

    xyzone = np.stack((x, y, z, np.ones_like(x)))

    xyzone = np.dot(M, xyzone)

    x = xyzone[0]
    y = xyzone[1]
    z = xyzone[2]

    lon = np.arctan(x/z)
    lat = np.arcsin(y/np.sqrt(x**2+y**2+z**2))

    lonlatone = np.stack((lon,lat,1))
    uvone = np.tensordot(P, lonlatone, axes=1)

    return (uvone[0],uvone[1])

def get_bbox2d_from_bbox3d(cam_proj_matrix, rx, ry, rz, sl, sh, sw, tx, ty, tz):

    bbox_to_velo_matrix = np.matmul(transformations.translation_matrix([tx, ty, tz]), transformations.euler_matrix(rx, ry, rz, 'sxyz'))
    bbox_proj_matrix = np.matmul(cam_proj_matrix, bbox_to_velo_matrix)

    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                v = np.array([(k-0.5)*sl, -(l)*sh, (m-0.5)*sw, 1.])
                v = np.matmul(bbox_proj_matrix, v)

                vertices[k,l,m,:] = spherical_project(v[0], v[1], v[2])

    vertices = vertices.astype(np.int32)

    x1 = np.amin(vertices[:,:,:,0])
    x2 = np.amax(vertices[:,:,:,0])
    y1 = np.amin(vertices[:,:,:,1])
    y2 = np.amax(vertices[:,:,:,1])

    return (x1,y1,x2,y2)

def model_matrix(root_dir, index):
    calib_path = os.path.join(root_dir, 'calib', index + ".txt")

    with open(calib_path,'r') as f:
        for line in f:
            fields = line.split(' ')
            if fields[0] == 'P2:':
                P = [float(s) for s in fields[1:]]
                P = np.reshape(P, [3,4])
                P[:2,:] /= P[0,0]
                P[:2,2] = 0
                return P

def get_image_path(root_dir, index):
    return os.path.join(root_dir, 'image_2_proj', index + ".png")

def generate_labels(image_set, subset, root_dir, output_filename):
    
    image_set_file = os.path.join(script_directory, 'ImageSets', image_set + '.txt')
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]

    output_file = open(output_filename, "w")

    for index in image_index:
        generate_image_labels(root_dir, subset, output_file, index)

def generate_image_labels(root_dir, subset, output_file, index):

    filename = os.path.join(root_dir, 'label_2', index + '.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    write_kitti_labels=False

    if write_kitti_labels:
        filename_kitti_proj_out = os.path.join(root_dir, 'label_2_proj', index + '.txt')
        file_kitti_proj = open(filename_kitti_proj_out, 'w')

    model_mat = model_matrix(root_dir, index)
    proj_mat = panoramic_projection_matrix()

    width = 469
    height = 168

    output_file.write('# %d\n' % int(index))
    output_file.write('%s\n' % get_image_path(root_dir, index))
    output_file.write('%d\n%d\n%d\n' % (3, height, width))

    output_file.write(" ".join([str(a) for a in model_mat.flatten()]))
    output_file.write(" 0.0 0.0 0.0 1.0")
    output_file.write('\n')
    output_file.write(" ".join([str(a) for a in proj_mat[:2].flatten()]))
    output_file.write('\n')

    object_lines = []
    dontcare_lines = []

    # Load object bounding boxes into a data frame.
    for line in lines:
        obj = line.strip().split(' ')

        unknownObject = False

        if subset == 'vehicle':
            if obj[0] == "Car":
                label = 1
            elif obj[0] == "Van":
                label = 2
            elif obj[0] == "Truck":
                label = 3
            elif obj[0] == "Tram" or obj[0] == "Misc":
                label = 4
            elif obj[0] != "DontCare":
                unknownObject = True
        elif subset == 'ped_cyc':
            if obj[0] == "Person_sitting":
                obj[0] = "DontCare"
            if obj[0] == "Pedestrian":
                label = 1
            elif obj[0] == "Cyclist":
                label = 2
            elif obj[0] != "DontCare":
                unknownObject = True
        elif subset == 'all':
            if obj[0] == "Car":
                label = 1
            elif obj[0] == "Van":
                label = 2
            elif obj[0] == "Truck":
                label = 3
            elif obj[0] == "Tram" or obj[0] == "Misc":
                label = 4
            elif obj[0] == "Person_sitting":
                obj[0] = "DontCare"
            elif obj[0] == "Pedestrian":
                label = 5
            elif obj[0] == "Cyclist":
                label = 6
            elif obj[0] != "DontCare":
                unknownObject = True

        # 0-based coordinates
        alpha = float(obj[3])
        #x1 = float(obj[4])
        #y1 = float(obj[5])
        #x2 = float(obj[6])
        #y2 = float(obj[7])
        ry = float(obj[14])
        h = float(obj[8])
        w = float(obj[9])
        l = float(obj[10])
        tx = float(obj[11])
        ty = float(obj[12])
        tz = float(obj[13])
        truncation = float(obj[1])
        occlusion = float(obj[2])
        
        x1,y1,x2,y2 = get_bbox2d_from_bbox3d(model_mat, 0, ry, 0, l, h, w, tx, ty, tz)
        x1 = max(0,min(x1,width))
        x2 = max(0,min(x2,width))
        y1 = max(0,min(y1,height))
        y2 = max(0,min(y2,height))
        
        if write_kitti_labels:
            file_kitti_proj.write("%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (obj[0], truncation, occlusion, alpha, x1, y1, x2, y2, h, w, l, tx, ty, tz, ry))

        if unknownObject:
            continue

        ignore = 0
        if (occlusion>=2 or truncation>=0.5):
            ignore = 1

        if obj[0] == "DontCare":
            dontcare_lines.append('%d %d %d %d\n' % (round(x1), round(y1), round(x2), round(y2)))
        else:
            object_lines.append('%d %d %d %d %d %d %f %f %f %f %f %f %f\n' % (label, ignore, round(x1), round(y1), round(x2), round(y2), h, w, l, tx, ty, tz, ry))
        
    output_file.write("%d\n" % len(object_lines))
    for line in object_lines:
        output_file.write(line)
    output_file.write("%d\n" % len(dontcare_lines))
    for line in dontcare_lines:
        output_file.write(line)


def create_window_file(image_set, subset):
    root_dir = sys.argv[1]
    filename = os.path.join(script_directory, "window_files", subset + "_" + image_set + ".txt")
    generate_labels(image_set, subset, root_dir, filename)

if __name__ == '__main__':
    image_sets = ['train', 'trainval', 'val']
    subsets = ['vehicle', 'ped_cyc', 'all']

    try:
        os.mkdir(os.path.join(script_directory,"window_files"))
    except:
        pass

    for image_set,subset in itertools.product(image_sets, subsets):
        create_window_file(image_set, subset)

