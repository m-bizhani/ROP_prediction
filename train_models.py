#%%
import os
import tensorflow as tf
tf.keras.backend.clear_session()
from models import *
from load_data import load_and_preprocess
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sns.set()

###Preamble 
batch_size = 128
tf.random.set_seed(1234)

###Load data

(X_train, y_train), (X_test, y_test) = load_and_preprocess(path1='./data/well1.csv', 
                                                            path2='./data/well2.csv')

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

###Creat tf.dataset for model training 
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


###Select the model number to load and train 

model_num = 1

if model_num==1:
    """
    Build and compile the deterministic model (Model # 1)
    """
    model = get_model1(input_shape=(X_train.shape[1], ))
    print(model.summary())
    
if model_num==2:
    """
    Build and compile model 2 (model with aleatoric uncertanity) 
    """
    model = get_model2(input_shape=(X_train.shape[1], ))
    print(model.summary())
    
if model_num==3:
    """
    Build and compile model 3 (model with epistemic uncertanity) 
    """
    model = get_model3(input_shape=(X_train.shape[1], ), X_train_size=X_train.shape[1])
    print(model.summary())
    
if model_num==4:
    """
    Build and compile model 4 (Bayesian neural network)
    """
    model = get_model4(input_shape=(X_train.shape[1], ), X_train_size=X_train.shape[1])
    print(model.summary())

#TODO create model checkpoints for saving weights and other callbacks 
ckp_path = '/'

cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1), 
    callbacks.ReduceLROnPlateau(patience=10, verbose=1, factor=0.5, min_lr=1e-7),
    callbacks.CSVLogger(ckp_path+ 'Model_' + str(model_num) +'_logs.csv', append=False),
    callbacks.ModelCheckpoint(ckp_path+ 'Model_' + str(model_num) +'.h5', 
                                save_best_only=True, verbose=1, save_weights_only=True)

]


###############Train the model%%%%%%%%%%%%%%
#TODO set number of epochs 

epochs = 1000

gpu = tf.test.gpu_device_name()

if gpu:
    print('GPU available \t training on GPU')
    
    with tf.device(gpu):
        hist = model.fit(train_ds, 
                        epochs=epochs, 
                        validation_data=test_ds, 
                        callbacks=cb,
                        verbose=1)
elif not gpu:
    print('No GPU available \t training on CPU')
    hist = model.fit(train_ds, 
                    epochs=epochs, 
                    validation_data=test_ds, 
                    callbacks=cb,
                    verbose=1)
    
    
#####Load the best model####
model = model.load_weights(ckp_path + 'Model_' + str(model_num) +'.h5')


####Plot learning curves

loss_vals = pd.DataFrame(hist.history)
fig, ax = plt.subplots(dpi=120)
ax.plot(loss_vals['mae'], label='Training data')
ax.plot(loss_vals['val_mae'], label='Validation data')
ax.set_ylabel('Mean absolute error')
ax.set_xlabel('Epoch')
ax.legend(frameon=False)
plt.show()