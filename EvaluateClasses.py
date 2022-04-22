import tensorflow as tf
import numpy as np
from Errors import compute_errors
from Loss import depth_loss_function
from dataset.kitti_dataset import KittiDatasetAugmented
from DataGenerator import DataGeneratorAugmented
from CityscapeClasses import CityscapeClasses, CityscapeClassNames
from Configuration import WIDTH, HEIGHT, BATCH_SIZE, NB_CLASSES

# Create an empty array for each class to hold the pixel values
classResult = []
for i in range(0, NB_CLASSES):
    classResult.append([])

def main():
    # Import both models
    model = tf.keras.models.load_model('model/model.tf', custom_objects = {"depth_loss_function": depth_loss_function})
    modelAug = tf.keras.models.load_model('model/modelAug.tf', custom_objects = {"depth_loss_function": depth_loss_function})
    
    # Import the dataset and create a generator
    kittiDataset = KittiDatasetAugmented("dataset/kitti/")
    valData = kittiDataset.load_val()

    validation_loader = DataGeneratorAugmented(
        data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    # For each input use both models to predict the output
    # Using the semantic segmentation group the results by class
    for i in range(0, validation_loader.__len__()):
        print(i, validation_loader.__len__())
        (x, y) = validation_loader.__getitem__(i)
        y_pred = model.predict(x[0])
        y_predAug = modelAug.predict(x)
        
        x_interp = np.interp(x, (0, 1), (0, 255))
        x_interp = x_interp.astype(np.uint16)
        
        # Associate each pixel value to its respective class
        for k in range(0, BATCH_SIZE):
            if(np.mean(x[0][k]) == 0): 
                continue
            for i in range(0, HEIGHT):
                for j in range(0, WIDTH):
                    try:
                        pixel_class = CityscapeClasses[tuple(x_interp[1][k][i][j])]
                        classResult[pixel_class[0]].append((y[k][i][j], y_pred[k][i][j], y_predAug[k][i][j]))
                    except:
                        print("Error")
                        pass
    
    print("Finished processing images")
    file = open("Class Evaluation Results.txt", "w")
    
    # Calculate the errors for each class and write to file   
    for k in range(0, NB_CLASSES):
        print("Class: " + str(k))
        if(len(classResult[k]) > 0):
            tmpClass = np.array_split(np.array(classResult[k]), 3, axis=1)
            gt = np.squeeze(tmpClass[0], axis=1)
            pred = np.squeeze(tmpClass[1], axis=1)
            predAug = np.squeeze(tmpClass[2], axis=1)
            
            gt = np.squeeze(gt, axis=1)
            pred = np.squeeze(pred, axis=1)
            predAug = np.squeeze(predAug, axis=1)
            
            file.write('Class: ' + str(k) + " " + CityscapeClassNames[k] + '\n')
            mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3 = compute_errors(gt, pred)
            file.write("Original Model\n")
            file.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format('mae', 'imae', 'abs_rel', 'sq_rel', 'mse', 'rmse', 'rmse_log', 'irmse', 'delta1', 'delta2', 'delta3'))
            file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}\n".format(mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3))

            mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3 = compute_errors(gt, predAug)
            file.write("\nAugmented Model\n")
            file.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format('mae', 'imae', 'abs_rel', 'sq_rel', 'mse', 'rmse', 'rmse_log', 'irmse', 'delta1', 'delta2', 'delta3'))
            file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}\n".format(mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3))
            file.write("\n")
    
    file.close()   
    
if __name__ == "__main__":
    main()
