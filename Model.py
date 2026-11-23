import os
import warnings
# Hide TensorFlow INFO/WARNING messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Optional: disable oneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")



import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import random
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#forgraph and matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
import numpy as np



#================================              Class               ===========================
class CNN_Model:
    def __init__(self, ntrain, input_size, batch_size, learning_rate):
        self.__ntrain = ntrain
        self.__input_size = input_size
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__classes = ['Blackseed', 'Diamond', 'Malagkit', 'R160', 'Triple2']
        self.__model_list = []
        self.__test_ds = None
        self.__fold_1 = None
        self.__fold_2 = None
        self.__fold_3 = None
        self.__fold_4 = None
        self.__fold_5 = None
        self.__last_conv_layer = "gradcam_conv"
        self.__histories = None
        self.__avg_f1_score = []

        
    #=============================              Load Kfold datasets          ================================
    def load_dataset(self):
        source = "./src/datasets/k-fold"

        #load test dataset
        test_path = os.path.join(source, "test")
        self.__test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            image_size = self.__input_size,
            batch_size = self.__batch_size
        ).map(lambda x, y: (x/255.0, y))


        #load fold 1
        fold1_path = os.path.join(source, "fold_1")
        fold_1 = tf.keras.utils.image_dataset_from_directory(
            fold1_path,
            image_size = self.__input_size,
            batch_size = self.__batch_size
        ).map(lambda x, y: (x/255.0, y))

        #load fold 2
        fold2_path = os.path.join(source, "fold_2")
        fold_2 = tf.keras.utils.image_dataset_from_directory(
            fold2_path,
            image_size = self.__input_size,
            batch_size = self.__batch_size
        ).map(lambda x, y: (x/255.0, y))

        #load fold 3
        fold3_path = os.path.join(source, "fold_3")
        fold_3 = tf.keras.utils.image_dataset_from_directory(
            fold3_path,
            image_size = self.__input_size,
            batch_size = self.__batch_size
        ).map(lambda x, y: (x/255.0, y))

        #load fold 4
        fold4_path = os.path.join(source, "fold_4")
        fold_4 = tf.keras.utils.image_dataset_from_directory(
            fold4_path,
            image_size = self.__input_size,
            batch_size = self.__batch_size
        ).map(lambda x, y: (x/255.0, y))

        #load fold 5
        fold5_path = os.path.join(source, "fold_5")
        fold_5 = tf.keras.utils.image_dataset_from_directory(
            fold5_path,
            image_size = self.__input_size,
            batch_size = self.__batch_size
        ).map(lambda x, y: (x/255.0, y))


        AUTOTUNE = tf.data.AUTOTUNE
        self.__fold_1 = fold_1.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.__fold_2 = fold_2.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.__fold_3 = fold_3.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.__fold_4 = fold_4.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.__fold_5 = fold_5.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)



        return print(
            f"Total test items: {sum(1 for _ in self.__test_ds)}\n"
            f"Total fold1 items: {sum(1 for _ in fold_1)}\n"
            f"Total fold2 items: {sum(1 for _ in fold_2)}\n"
            f"Total fold3 items: {sum(1 for _ in fold_3)}\n"
            f"Total fold4 items: {sum(1 for _ in fold_4)}\n"
            f"Total fold5 items: {sum(1 for _ in fold_5)}"
        )


        #===========================                   Pipeline                ===========================================

    def pipeline(self):

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.00005),
            layers.RandomBrightness(0.00005),
        ])
        
        self.__model = models.Sequential([
            data_augmentation,
    
            #convolutional pipeline
            
            # featuremap formula = output(input-kernel) / stride + 1
            # create 16 featuremaps with 3x3 kernel and 2x2 window
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.__input_size, 3)),
            layers.MaxPooling2D((2, 2)),  # perform maxpool
            layers.BatchNormalization(),


            # create 128 featuremaps with 3x3 window
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),  # perform maxpool
            layers.BatchNormalization(),

                # create 128 featuremaps with 3x3 window
            layers.Conv2D(64, (3, 3), activation='relu', name=self.__last_conv_layer),
            layers.MaxPooling2D((2, 2)),  # perform maxpool
            layers.BatchNormalization(),
            
                 # create 128 featuremaps with 3x3 window
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),  # perform maxpool
            layers.BatchNormalization(),

    

            # Flatten: converts 3D feature maps into a 1D vector for dense layers.
            #layers.Flatten(),
            layers.GlobalAveragePooling2D(),
    
            # hidden layer
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),  # Prevents overfitting by forcing the network to generalize.
    
            # output layer
            layers.Dense(len(self.__classes), activation='softmax')
        ])
                    
            

                
                        #compile
        self.__model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = self.__learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"]
            )
        
        #self.__model.summary()

        


        
    #================================             Train Model            =================================
    #TRAIN MODEL WITH 5 FOLD OR 5 K 
    def train(self):
                         
        k = 5
        fold_accuracies = []  #STORE EACH FOLDS ACCURACIES
        self.__histories = []

        #LOOP IN A LENGTH OF K's
        for i in range(k):
            self.pipeline() 

            #callbacks
            checkpoint_cb = ModelCheckpoint(
                filepath = f"./src/saved_model/best_model_fold_{i+1}.keras",       
                save_best_only=True,            
                monitor="val_loss",              
                mode="min",                     
                verbose=1
            )
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-5
            )

            
            val_ds = None
            train_ds = None
            if i == 0:
                val_ds = self.__fold_1
                train_ds = self.__fold_2.concatenate(self.__fold_3).concatenate(self.__fold_4).concatenate(self.__fold_5)
            elif i == 1:
                val_ds = self.__fold_2
                train_ds = self.__fold_1.concatenate(self.__fold_3).concatenate(self.__fold_4).concatenate(self.__fold_5)
            elif i == 2:
                val_ds = self.__fold_3
                train_ds = self.__fold_1.concatenate(self.__fold_2).concatenate(self.__fold_4).concatenate(self.__fold_5)
            elif i == 3:
                val_ds = self.__fold_4
                train_ds = self.__fold_1.concatenate(self.__fold_2).concatenate(self.__fold_3).concatenate(self.__fold_5)
            elif i == 4:
                val_ds = self.__fold_5
                train_ds = self.__fold_1.concatenate(self.__fold_2).concatenate(self.__fold_3).concatenate(self.__fold_4)


            
            print(f"\n==========   Training fold {i+1}   ==========")
            history = self.__model.fit(
                train_ds,
                validation_data=val_ds,
                initial_epoch=0,  #starting chechpoint
                epochs=self.__ntrain,          
                callbacks=[checkpoint_cb, early_stop, reduce_lr]
            )

            self.__histories.append(history)
            self.__model_list.append(self.__model)
            
            best_epoch = np.argmin(history.history['val_loss'])
            val_accuracy = history.history['val_accuracy'][best_epoch]
            fold_accuracies.append(val_accuracy)



            self.__model.save(f"./src/saved_model/fold_{i+1}_model.keras")
            
        # summary
        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print("\n=== K-Fold Results With Lowest Val Loss===")
        for i, acc in enumerate(fold_accuracies, 1):
            print(f"Fold {i}: {acc:.4f}")
        print(f"Average validation accuracy: {avg_accuracy:.4f}")
        
        return self.__model


        

#===============================                   Evaluate Model                         ====================================
    
    def smooth_curve(self, points, factor=0.9):
        smoothed = []
        for p in points:
            if smoothed:
                smoothed.append(smoothed[-1] * factor + p * (1 - factor))
            else:
                smoothed.append(p)
        return smoothed

    def eval(self):
        plt.figure(figsize=(11,3), dpi=250)
    
        for i, h in enumerate(self.__histories, 1):
            raw_vals = h.history['val_accuracy']
            smoothed_vals = self.smooth_curve(raw_vals, factor=0.1)  # light smoothing
        
            line, = plt.plot(smoothed_vals, label=f'Fold {i}')
        
            best_epoch = np.argmin(h.history['val_loss'])
            best_acc = raw_vals[best_epoch]
            plt.scatter(best_epoch, best_acc, color=line.get_color(), marker='o')
            plt.text(best_epoch, best_acc+0.005, f"{best_acc:.2f}", color=line.get_color())




    
        plt.title("Validation Accuracy Progression")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
    
        fold_accuracies = [
            h.history['val_accuracy'][np.argmin(h.history['val_loss'])]
            for h in self.__histories
        ]
        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        
        # Draw bars for each fold
        plt.bar(range(1, len(fold_accuracies)+1), fold_accuracies, color="skyblue")
        
        # Draw average line
        plt.axhline(avg_accuracy, color='red', linestyle='--', label=f'Average = {avg_accuracy:.4f}')
        
        # Annotate each bar with its value
        for i, acc in enumerate(fold_accuracies, 1):
            plt.text(i, acc+0.005, f"{acc:.4f}", ha='center', color='blue')
        
        plt.xlabel("Fold")
        plt.ylabel("Best Validation Accuracy")
        plt.legend()
        plt.ylim(0.5, 1.0)
        plt.show()

        #test eval
        for i in range(len(self.__model_list)):
            
            #test accuraccy
            fold_model = self.__model_list[i]
            test_loss, test_acc = fold_model.evaluate(self.__test_ds, verbose=1)
            print("==================================================================")
            print("                         Test Accuracy                            ")
            print(f"\n[Fold {i+1}]")
            print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
            print("==================================================================")


        
            #Predictions and F1 scores
            y_true = []
            y_pred = []

            for images, labels in self.__test_ds:
                predictions = fold_model.predict(images, verbose = 0)
                y_true.extend(labels.numpy())
                y_pred.extend(np.argmax(predictions, axis=1))

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')

            print("Classification Result")
    
            print(classification_report(
                y_true,
                y_pred,
                target_names=self.__classes,
                digits=4
            ))

    #================================        Display test datasets images with result            ====================================
    def display_test_results(self, batches):

        for images, labels in self.__test_ds.take(batches):
    
            plt.figure(figsize=(16, 12), dpi=500)
    
            for i in range(min(12, len(images))):
    
                ax = plt.subplot(3, 4, i + 1)
    
                img = images[i]
                true_label = labels[i].numpy()
    
                # Predict image
                prediction = self.__model.predict(
                    np.expand_dims(img.numpy(), axis=0),
                    verbose=0
                )
    
                pred_class = np.argmax(prediction[0])
    
                #class names
                true_class = self.__classes[true_label]
                predicted_class = self.__classes[pred_class]
    
                # Display image
                plt.imshow(img.numpy())
    
                # Green = correct, Red = incorrect
                if pred_class == true_label:
                    color = "green"
                else:
                    color = "red"
    
                plt.title(
                    f"P: {predicted_class}\n"
                    f"A: {true_class}",
                    color=color
                )
    
                plt.axis("off")
    
            plt.tight_layout()
            plt.show()
            
       


    #================================                     Gradcam function              ======================================
    
    def gradcam_heatmap(self, img_array, model, last_conv_layer, pred_index=None):
    
        with tf.GradientTape() as tape:
    
            x = img_array
            conv_outputs = None
    
            #Forward pass through the actual model layers
            for layer in self.__model.layers:
    
                x = layer(x, training=False)
    
                #Save the output of the target convolutional layer
                if layer.name == last_conv_layer:
                    conv_outputs = x
    
            #Final model prediction
            predictions = x
    
            #If no class was specified, use predicted class
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
    
            #Score for the target class
            class_score = predictions[:, pred_index]
    
        #Gradient of class score with respect to feature maps
        grads = tape.gradient(
            class_score,
            conv_outputs
        )
    
        #Make sure gradients exist
        if grads is None:
            raise ValueError(
                "Gradients are None. Check the Grad-CAM target layer."
            )
    
        #Average gradients across height and width
        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )
    
        # emove batch dimension
        conv_outputs = conv_outputs[0]
    
        #Weight each feature map by its importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    
        heatmap = tf.squeeze(heatmap)
      
        #Keep only positive values
        heatmap = tf.maximum(
            heatmap,
            0
        )
    
        #Normalize between 0 and 1
        heatmap /= (
            tf.reduce_max(heatmap)
            + tf.keras.backend.epsilon()
        )
    
        return heatmap.numpy()





    
    #=================================            display grad_cam heatmap           ===========================
    def gradcam(self, fold):
        rand_class =  random.choice(self.__classes)
        image_path = f"./src/datasets/k-fold/{fold}/{rand_class}"
        images = os.listdir(image_path)
        rand_choice = random.choice(images)
        image = os.path.join(image_path, rand_choice)
        image = tf.keras.utils.load_img(
            image,
            target_size = self.__input_size
        )
        
        image_array = tf.keras.utils.img_to_array(image)
        image_array = image_array/255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = self.__model.predict(image_array)
        predicted_index = np.argmax(prediction[0])
        predicted_class = self.__classes[predicted_index]

        #img_array, model, last_conv_layer,
        heatmap = self.gradcam_heatmap(image_array, self.__model, self.__last_conv_layer, pred_index=predicted_index)


        original_image = tf.keras.utils.img_to_array(image)
        heatmap_resize = tf.image.resize(
            heatmap[..., np.newaxis],
            self.__input_size
        )

        heatmap_resize = tf.squeeze(
            heatmap_resize
        ).numpy()

        #Display graph 
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image.astype("uint8"))

        plt.title(
            f"Original\n"
            f"Actual: {rand_class}"
        )

        plt.axis("off")

        # Grad-CAM overlay
        plt.subplot(1, 2, 2)
        
        plt.imshow(
            original_image.astype("uint8")
        )
        
        plt.imshow(
            heatmap_resize,
            cmap="jet",
            alpha=0.4
        )
        
        plt.title(
            f"Grad-CAM\n"
            f"Predicted: {predicted_class}\n"
        )

        print("Actual class:", f"{rand_class}"),
        print("Predicted class:", predicted_class)
        return {
            "actual_class": rand_class,
            "predicted_class": predicted_class,
            "heatmap": heatmap_resize
        }


        
#=============================                Main                  ==================================
#params <N_Training>, <input_size>, <batch_size>, <learning rate>
model = CNN_Model(100, (256, 256), 32, 5e-4)
model.load_dataset()
model.train()
