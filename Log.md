# This contains the log of my diploma thesis
## First Step:
Download the dataset and prepare an import script for it.

I downloaded:
- Raw Data: http://www.cvlibs.net/datasets/kitti/raw_data.php
- Annotated Depth Maps: http://www.cvlibs.net/datasets/kitti/eval_depth_all.php

The import scrip used the depth maps for image_02, which is the center RGB camera, in the image format.

___
## Second Step:
Use the base model from: https://keras.io/examples/vision/depth_estimation/

Try to train the data using the above model and the dataset.

The problem was that the given depth maps are sparse and contain depth information only for a small part of the pixels in the original image.

___
## Third Step:
Inspired from: https://github.com/ialhashim/DenseDepth

I generated the dense depth maps with a custom script that performed depth impainting. (Base: https://gist.github.com/ialhashim/be6235489a9c43c6d240e8331836586a)

___
## Fourth Step:
Using DenseDepth model from the third step and the model from the Second Step I created a custom cleaner model that allowed me to train a basic model.

___
## Fifth Step:
Pre-generate semantic segmented images using ERFNet (https://github.com/Eromera/erfnet_pytorch)

___
## Sixth Step:
Modify DataGenerator into a separate class. Create new augmented DataGenerator and KittiDataset classes that extend the original classes.

___
## Seventh Step:
Create the new augumented depth estimation model based on the original depth estimation model.

___
## Eighth Step:
Evaluation scripts done. Evaluation can be done overall or by classes.

___
## Nineth Step:
Changed configuration for better training and exported initial results.

___
## TO DO:
- Train a better model (High resolution training)
- Second Augumented Depth Estimation Model?