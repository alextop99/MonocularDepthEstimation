from dataset.kitti_dataset import KittiDatasetAugmented
import tensorflow as tf
from Loss import depth_loss_function
from DataGenerator import DataGeneratorAugmented
from Models import DepthEstimationAugmentedModel
from Configuration import WIDTH, HEIGHT, BATCH_SIZE, LR, EPOCHS

tf.random.set_seed(123)

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR,
        amsgrad=False,
    )
    
    model = DepthEstimationAugmentedModel()
    
    # Compile the model
    model.compile(optimizer, loss=depth_loss_function)

    kittiDataset = KittiDatasetAugmented("dataset/kitti/")
    trainData, valData = kittiDataset.load_train_val()

    train_loader = DataGeneratorAugmented(
        data=trainData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    validation_loader = DataGeneratorAugmented(
        data=valData, batch_size=BATCH_SIZE, dim=(WIDTH, HEIGHT)
    )
    
    checkpoint_path = "checkpointsAug/cpAug.ckpt"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)
    
    
    model.fit(
        train_loader,
        epochs=EPOCHS,
        validation_data=validation_loader,
        callbacks=[cp_callback]
    )
    
    model.save_weights("model/weightsAug.h5")
    model.save("model/modelAug.tf", save_format='tf')

if __name__ == "__main__":
    main()