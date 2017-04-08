import sys
import os
import math
import numpy as np
import itertools

script_directory = os.path.dirname(os.path.realpath(sys.argv[0]))

def projection_matrix(root_dir, index):
    calib_path = os.path.join(root_dir, 'calib', index + ".txt")

    with open(calib_path,'r') as f:
        for line in f:
            fields = line.split(' ')
            if fields[0] == 'P2:':
                return [float(s) for s in fields[1:]]

def get_image_path(root_dir, index):
    return os.path.join(root_dir, 'image_2', index + ".png")

def generate_labels(image_set, subset, root_dir, output_filename):

    image_set_file = os.path.join(script_directory, 'ImageSets', image_set + '.txt')
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]

    output_file = open(output_filename, "w")

    for index in image_index:
        generate_image_labels(root_dir, subset, output_file, index)

def generate_image_labels(root_dir, subset, output_file, index):
    """
    Load image and bounding boxes info from txt file in the KITTI
    format.
    """

    filename = os.path.join(root_dir, 'label_2', index + '.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    proj_mat = projection_matrix(root_dir, index)

    width = 1242
    height = 375

    output_file.write('# %d\n' % int(index))
    output_file.write('%s\n' % get_image_path(root_dir, index))
    output_file.write('%d\n%d\n%d\n' % (3, height, width))

    output_file.write(" ".join([str(a) for a in proj_mat]))
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
        x1 = float(obj[4])
        y1 = float(obj[5])
        x2 = float(obj[6])
        y2 = float(obj[7])
        ry = float(obj[14])
        h = float(obj[8])
        w = float(obj[9])
        l = float(obj[10])
        tx = float(obj[11])
        ty = float(obj[12])
        tz = float(obj[13])
        truncation = float(obj[1])
        occlusion = float(obj[2])

        
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

