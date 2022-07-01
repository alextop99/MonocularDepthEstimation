import tensorflow as tf
import numpy as np
from Errors import compute_errors
from Loss import depth_loss_function
from dataset.kitti_dataset import KittiDatasetAugmented
from DataGenerator import DataGeneratorAugmented
from CityscapeClasses import CityscapeClasses, CityscapeClassNames
from Configuration import WIDTH, HEIGHT, BATCH_SIZE, NB_CLASSES
import gc

# Create an empty array for each class to hold the pixel values
perClassGt = []
perClassPred = []
perClassPredAug = []
for i in range(0, NB_CLASSES):
    perClassGt.append([])
    perClassPred.append([])
    perClassPredAug.append([])
    
results = []
resultsAug = []
for i in range(0, NB_CLASSES):
    results.append([])
    resultsAug.append([])
    for j in range(0, 11):
        results[i].append([])
        resultsAug[i].append([0])
        
apparitions = [False] * NB_CLASSES

def main():
    global perClassGt, perClassPred, perClassPredAug
    # Import both models
    model = tf.keras.models.load_model('model/model.tf', custom_objects = {"depth_loss_function": depth_loss_function})
    modelAug = tf.keras.models.load_model('model/modelAug.tf', custom_objects = {"depth_loss_function": depth_loss_function})
    
    # Import the dataset and create a generator
    kittiDataset = KittiDatasetAugmented("dataset/kitti/")
    valData = kittiDataset.load_val()

    validation_loader = DataGeneratorAugmented(
        data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT), evaluate=True
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
        for batch in range(0, BATCH_SIZE):
            if(np.mean(x[0][batch]) == 0): 
                continue
            for height in range(0, HEIGHT):
                for width in range(0, WIDTH):
                    pixel_class = CityscapeClasses[tuple(x_interp[1][batch][height][width])]
                    apparitions[pixel_class[0]] = True
                    perClassGt[pixel_class[0]].append(y[batch][height][width])
                    perClassPred[pixel_class[0]].append(y_pred[batch][height][width])
                    perClassPredAug[pixel_class[0]].append(y_predAug[batch][height][width])
        
        for semClass in range(0, NB_CLASSES):
            if(len(perClassGt[semClass]) > 0):
                gt = np.array(perClassGt[semClass])
                pred = np.array(perClassPred[semClass])
                predAug = np.array(perClassPredAug[semClass])
                
                mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3 = compute_errors(gt, pred)
                
                if mae != np.nan:
                    results[semClass][0].append(mae)
                    results[semClass][1].append(imae)
                    results[semClass][2].append(abs_rel)
                    results[semClass][3].append(sq_rel)
                    results[semClass][4].append(mse)
                    results[semClass][5].append(rmse)
                    results[semClass][6].append(rmse_log)
                    results[semClass][7].append(irmse)
                    results[semClass][8].append(delta1)
                    results[semClass][9].append(delta2)
                    results[semClass][10].append(delta3)
                
                mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3 = compute_errors(gt, predAug)
                
                if mae != np.nan:
                    resultsAug[semClass][0].append(mae)
                    resultsAug[semClass][1].append(imae)
                    resultsAug[semClass][2].append(abs_rel)
                    resultsAug[semClass][3].append(sq_rel)
                    resultsAug[semClass][4].append(mse)
                    resultsAug[semClass][5].append(rmse)
                    resultsAug[semClass][6].append(rmse_log)
                    resultsAug[semClass][7].append(irmse)
                    resultsAug[semClass][8].append(delta1)
                    resultsAug[semClass][9].append(delta2)
                    resultsAug[semClass][10].append(delta3)
        
        del perClassGt
        del perClassPred
        del perClassPredAug
        gc.collect()
        
        perClassGt = []
        perClassPred = []
        perClassPredAug = []
        for i in range(0, NB_CLASSES):
            perClassGt.append([])
            perClassPred.append([])
            perClassPredAug.append([])
                    
        
    
    print("Finished processing images")
    file = open("Class Evaluation Results.txt", "w")
    for semClass in range(0, NB_CLASSES):
        if(len(results[semClass][0]) > 0):
            file.write('Class: ' + str(semClass) + " " + CityscapeClassNames[semClass] + '\n')
            file.write("Original Model\n")
            file.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format('mae', 'imae', 'abs_rel', 'sq_rel', 'mse', 'rmse', 'rmse_log', 'irmse', 'delta1', 'delta2', 'delta3'))
            file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}\n".format(
                np.array(results[semClass][0]).mean(), np.array(results[semClass][1]).mean(), np.array(results[semClass][2]).mean(), np.array(results[semClass][3]).mean(), 
                np.array(results[semClass][4]).mean(), np.array(results[semClass][5]).mean(), np.array(results[semClass][6]).mean(), np.array(results[semClass][7]).mean(), 
                np.array(results[semClass][8]).mean(), np.array(results[semClass][9]).mean(), np.array(results[semClass][10]).mean()))

            file.write("\nAugmented Model\n")
            file.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format('mae', 'imae', 'abs_rel', 'sq_rel', 'mse', 'rmse', 'rmse_log', 'irmse', 'delta1', 'delta2', 'delta3'))
            file.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}\n".format(
                np.array(resultsAug[semClass][0]).mean(), np.array(resultsAug[semClass][1]).mean(), np.array(resultsAug[semClass][2]).mean(), np.array(resultsAug[semClass][3]).mean(), 
                np.array(resultsAug[semClass][4]).mean(), np.array(resultsAug[semClass][5]).mean(), np.array(resultsAug[semClass][6]).mean(), np.array(resultsAug[semClass][7]).mean(), 
                np.array(resultsAug[semClass][8]).mean(), np.array(resultsAug[semClass][9]).mean(), np.array(resultsAug[semClass][10]).mean()))
            file.write("\n") 
    
    file.close()
    
    for x in apparitions:
        print(x)
    
if __name__ == "__main__":
    main()
