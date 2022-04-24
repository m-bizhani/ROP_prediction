from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, optimizers, layers, callbacks, activations, models, initializers, constraints, initializers, regularizers, losses
import tensorflow as tf
import tensorflow_probability as tfp
from common import _mae_mse, nll, r2

K.clear_session()

tfpl = tfp.layers
tfd = tfp.distributions

def get_model1(input_shape):
    """Instantiate and complies model 1

    Args:
        input_shape (int): input shape (i.e. number of features)

    Returns:
        model object 
    """
    k_init = initializers.GlorotUniform(seed=1234)
    
    x_in = Input(input_shape)
    
    x = layers.Dense(512*4, kernel_initializer=k_init, kernel_regularizer=regularizers.L2())(x_in)
    x = activations.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(256, kernel_initializer=k_init, kernel_regularizer=regularizers.L2())(x)
    x = activations.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512*4, kernel_initializer=k_init)(x)
    x = activations.relu(x)
    
    x_o = layers.Dense(1, activation=None, kernel_constraint=constraints.NonNeg())(x)
    model = Model(x_in, x_o, name='Model_1')
    
    model.compile(optimizer = optimizers.Adam(1e-4), 
            loss = _mae_mse, 
            metrics = ['mse', 'mae', r2])
    
    return model



def get_model2(input_shape):
    """Instantiate and complies model 2

    Args:
        input_shape (int): input shape (i.e. number of features)

    Returns:
        model object 
    """
    tf.keras.backend.clear_session()
    k_init = initializers.GlorotUniform(seed=1234)
    x_in = Input(input_shape)
    
    x = layers.Dense(512*4, kernel_initializer=k_init, kernel_regularizer=regularizers.L2())(x_in)
    x = activations.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(256, kernel_initializer=k_init, kernel_regularizer=regularizers.L2())(x)
    x = activations.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512*4, kernel_initializer=k_init)(x)
    x = activations.relu(x)

    x = layers.Dense(tfpl.IndependentNormal.params_size(event_shape=1))(x)
    x_o = tfpl.IndependentNormal(1, convert_to_tensor_fn=tfp.distributions.Distribution.sample)(x)

    model = Model(x_in, x_o, name='Model_2')
    
    model.compile(optimizer = optimizers.Adam(1e-4), 
            loss = nll, 
            metrics = ['mse', 'mae', r2])
    
    return model

def get_model3(input_shape, X_train_size):
    """Instantiate and complies model 3

    Args:
        input_shape (int): input shape 
        X_train_size (int): Number of examples in the training dataset (required for calculating KL divergence)

    Returns:
        model object
    """
    divergence_fn =  lambda q,p,_ : tfd.kl_divergence(q,p) / X_train_size
    def custome_layer(units=512*4):
        l  = tfpl.DenseReparameterization(units,
                                        kernel_posterior_fn=tfpl.util.default_mean_field_normal_fn(is_singular=False), 
                                        bias_prior_fn = tfpl.default_multivariate_normal_fn,
                                        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
                                        kernel_divergence_fn = divergence_fn,
                                        bias_divergence_fn = divergence_fn
                                        ) 
        return l                                                                             

    x_in = Input(shape=input_shape)
    x = custome_layer(512*4)(x_in)
    x = activations.swish(x)
    x = custome_layer(256)(x)
    x = activations.swish(x)
    x = custome_layer(512*4)(x)
    x = activations.swish(x)
    x = layers.Flatten()(x)
    x_o = layers.Dense(1, activation=None, kernel_constraint=constraints.NonNeg())(x)

    model = Model(x_in, x_o, name='Model_3')
    
    model.compile(optimizer = optimizers.Adam(1e-4), 
            loss = _mae_mse, 
            metrics = ['mse', 'mae', r2])
    
    return model



def get_model4(input_shape, X_train_size):
    """Instantiate and complies model 3

    Args:
        input_shape (int): input shape 
        X_train_size (int): Number of examples in the training dataset (required for calculating KL divergence)

    Returns:
        model object 
    """
    divergence_fn =  lambda q,p,_ : tfd.kl_divergence(q,p) / X_train_size
    def custome_layer(units=512*4):
        l  = tfpl.DenseReparameterization(units,
                                        kernel_posterior_fn=tfpl.util.default_mean_field_normal_fn(is_singular=False), 
                                        bias_prior_fn = tfpl.default_multivariate_normal_fn,
                                        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
                                        kernel_divergence_fn = divergence_fn,
                                        bias_divergence_fn = divergence_fn
                                        ) 
        return l                                                                             

    x_in = Input(shape=input_shape)
    x = custome_layer(512*4)(x_in)
    x = activations.swish(x)
    x = custome_layer(256)(x)
    x = activations.swish(x)
    x = custome_layer(512*4)(x)
    x = activations.swish(x)
    x = custome_layer(tfpl.IndependentNormal.params_size(1))(x)
    x_o = tfpl.IndependentNormal(1, convert_to_tensor_fn=tfp.distributions.Distribution.mean)(x)

    model = Model(x_in, x_o, name='Model_4')
    model.compile(optimizer = optimizers.Adam(1e-4), 
            loss = nll, 
            metrics = ['mse', 'mae', r2])
    return model

if __name__=='__main__':
    """test cell"""
    model1 = get_model1(input_shape=(10, ))
    print(model1.summary())
