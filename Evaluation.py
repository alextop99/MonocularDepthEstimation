import tensorflow as tf
import numpy as np
from Errors import compute_errors
from Loss import depth_loss_function
from dataset.kitti_dataset import KittiDataset, KittiDatasetAugmented
from DataGenerator import DataGenerator, DataGeneratorAugmented

WIDTH = 256
HEIGHT = 64

BATCH_SIZE = 6

def main():
    model = tf.keras.models.load_model('model/model.tf', custom_objects = {"depth_loss_function": depth_loss_function})
    modelAug = tf.keras.models.load_model('model/modelAug.tf', custom_objects = {"depth_loss_function": depth_loss_function})
    
    kittiDataset = KittiDatasetAugmented("dataset/kitti/")
    valData = kittiDataset.load_val()

    validation_loader = DataGeneratorAugmented(
        data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    maeArr = imaeArr = abs_relArr = sq_relArr = mseArr = rmseArr = rmse_logArr = irmseArr = delta1Arr = delta2Arr = delta3Arr = np.array([])
    maeAugArr = imaeAugArr = abs_relAugArr = sq_relAugArr = mseAugArr = rmseAugArr = rmse_logAugArr = irmseAugArr = delta1AugArr = delta2AugArr = delta3AugArr = np.array([])
    
    for i in range(0, validation_loader.__len__()):
        print(i, validation_loader.__len__())
        (x, y) = validation_loader.__getitem__(i)
        y_pred = model.predict(x[0])
        y_predAug = modelAug.predict(x)
        
        for j in range(0, BATCH_SIZE):
            mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3 = compute_errors(y[j], y_pred[j])
            maeArr = np.append(maeArr, mae)
            imaeArr = np.append(imaeArr, imae)
            abs_relArr = np.append(abs_relArr, abs_rel)
            sq_relArr = np.append(sq_relArr, sq_rel)
            mseArr = np.append(mseArr, mse)
            rmseArr = np.append(rmseArr, rmse)
            rmse_logArr = np.append(rmse_logArr, rmse_log)
            irmseArr = np.append(irmseArr, irmse)
            delta1Arr = np.append(delta1Arr, delta1)
            delta2Arr = np.append(delta2Arr, delta2)
            delta3Arr = np.append(delta3Arr, delta3)

            mae, imae, abs_rel, sq_rel, mse, rmse, rmse_log, irmse, delta1, delta2, delta3 = compute_errors(y[j], y_predAug[j])
            maeAugArr = np.append(maeAugArr, mae)
            imaeAugArr = np.append(imaeAugArr, imae)
            abs_relAugArr = np.append(abs_relAugArr, abs_rel)
            sq_relAugArr = np.append(sq_relAugArr, sq_rel)
            mseAugArr = np.append(mseAugArr, mse)
            rmseAugArr = np.append(rmseAugArr, rmse)
            rmse_logAugArr = np.append(rmse_logAugArr, rmse_log)
            irmseAugArr = np.append(irmseAugArr, irmse)
            delta1AugArr = np.append(delta1AugArr, delta1)
            delta2AugArr = np.append(delta2AugArr, delta2)
            delta3AugArr = np.append(delta3AugArr, delta3)
    
    print("\nOriginal Model")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('mae', 'imae', 'abs_rel', 'sq_rel', 'mse', 'rmse', 'rmse_log', 'irmse', 'delta1', 'delta2', 'delta3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(maeArr.mean(), imaeArr.mean(), abs_relArr.mean(), sq_relArr.mean(), mseArr.mean(), rmseArr.mean(), rmse_logArr.mean(), irmseArr.mean(), delta1Arr.mean(), delta2Arr.mean(), delta3Arr.mean()))
    
    print("\nAugmented Model")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('mae', 'imae', 'abs_rel', 'sq_rel', 'mse', 'rmse', 'rmse_log', 'irmse', 'delta1', 'delta2', 'delta3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(maeAugArr.mean(), imaeAugArr.mean(), abs_relAugArr.mean(), sq_relAugArr.mean(), mseAugArr.mean(), rmseAugArr.mean(), rmse_logAugArr.mean(), irmseAugArr.mean(), delta1AugArr.mean(), delta2AugArr.mean(), delta3AugArr.mean()))
    
if __name__ == "__main__":
    main()
