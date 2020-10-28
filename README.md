# Eliminating the Blind Spot: Adapting 3D Object Detection and Monocular Depth Estimation to 360° Panoramic Imagery
by Grégoire Payen de La Garanderie, Amir Atapour Abarghouei and Toby P. Breckon.

This is an implementation of the object detection approach presented in our paper on object detection and monocular depth estimation for 360° panoramic imagery. See our [project page](https://gdlg.github.io/panoramic) for more details. This implementation is based on the implementation of MS-CNN by Zhaowei Cai available [here](https://github.com/zhaoweicai/mscnn) and the deep learning framework [Caffe](https://github.com/BVLC/caffe).

## Citations
If you use our code, please cite our paper:

Eliminating the Dreaded Blind Spot: Adapting 3D Object Detection and Monocular Depth Estimation to 360° Panoramic Imagery
G. Payen de La Garanderie, A. Atapour Abarghouei, T.P. Breckon
*In Proc. European Conference on Computer Vision, Springer, 2018. (to appear)*

## Installation

Clone this repository then build the code using CMake:

```
mkdir build
cd build
cmake ..
make -j8
```

This code base was tested using CUDA 9.0, CuDNN 7.0 and OpenCV 3.4.0.

## Inference

Our inference script is based on panoramas that have been cropped vertically due to memory contraints. To use the same crop as in our paper, take a 2048×1024 panorama and crop it vertically to the range [424,724] to get a 2048x300 panorama.

1. Download a dataset of 360° images such as [our dataset](https://hochet.info/~gregoire/eccv2018/datasets/synthetic-panoramic-dataset-v2.tar.gz) of synthetic images based on the CARLA simulator. You can also use images from Mapillary (see the [list](https://hochet.info/~gregoire/models/mapillary_image_keys.txt) of Mapillary image keys that we used for experimentation).

*Note that the dataset was updated on the 25/02/2020 to improve the ground truth bounding box quality and add 3D object detection evaluation metrics. The original dataset is still available [here](https://hochet.info/~gregoire/eccv2018/datasets/synthetic-panoramic-dataset.tar.gz).*

2. (optional). Fetch our pretrained models:
```
bash ./download_models.sh
```

3. Run the inference script `detection.py`. Here is an example using our pretrained model on the CARLA dataset from step 1.
```
python2 detect.py --weights=models/360nn/mixed.caffemodel --input=~/data/carla-dataset/val/image --output=~/data/carla-example-detections
```

It is also possible to run the models on rectilinear images from the KITTI dataset:
```
python2 detect.py --weights=models/360nn/kitti.caffemodel --input=~/data/kitti/object/training/image --output=~/data/kitti-example-detections --model=examples/inference/deploy_rectilinear.prototxt --rectilinear
```

## Training

Training is based on a modified version of the KITTI dataset using style transformation. We provide the CycleGAN models that we have used to generate the transformed images. You can generate the style-transferred images using the following:

```
git clone https://github.com/gdlg/pytorch-CycleGAN-and-pix2pix
wget https://hochet.info/~gregoire/models/kitti_cyclegan_models.tar.gz
tar xzf kitti_cyclegan_models.tar.gz
python test_single.py --dataroot <path-to-kitti>/object/training/image_2 --results_dir <path-to-kitti>/object/training/image_2_miami --name miami-kitti  --model cycle_gan --phase train --no_dropout --resize_or_crop none --no_flip --dataset_mode single --which_direction=BtoA --which_epoch 4 --how_many 1000000
python test_single.py --dataroot <path-to-kitti>/object/training/image_2 --results_dir <path-to-kitti>/object/training/image_2_carla --name carla-kitti  --model cycle_gan --phase train --no_dropout --resize_or_crop none --no_flip --dataset_mode single --which_direction=BtoA --which_epoch 4 --how_many 1000000
```

Once those images have been generated, you can generate the ground truth data files and then train the object detection model using:
```
cd data
bash ./create_labels.sh
cd examples/training/mixed
bash ./train.sh
```

## License

Our contributions are released under the MIT license. 
The original Caffe code is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE), contributions of Faster R-CNN are release under the [MIT license](https://github.com/BVLC/caffe/blob/master/LICENSE) and the contributions from MS-CNN are released under a [custom license](https://github.com/zhaoweicai/mscnn/blob/master/%20LICENSE). Please see the LICENSE file for more details.

