
from tensorflow.keras import backend as K
from tensorflow.keras import losses
import tensorflow as tf

K.clear_session()


def r2(y_true, y_pred):
    """This utility function computes the R2 metric for monitoring model training

    Returns:
        R2 value 
    """
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def _mae_mse(y_true, y_pred, w1 = 1.0, w2 = 5.0):
    """Weighted MES and MAE loss 

    Args:
        y_true (_type_): True target value
        y_pred (_type_): Predeicted target value
        w1 (float, optional): Weight given to MSE loss. Defaults to 1.0.
        w2 (float, optional): Weight given to MAE loss. Defaults to 5.0.

    Returns:
        Weighted MAE-MSE loss
    """
    l1 = losses.mean_squared_error(y_true, y_pred)
    l2 = losses.mean_absolute_error(y_true, y_pred)
    
    return w1*l1 + w2*l2

def nll(y_true, y_pred):
    """Negative log-likelihood loss function
    """
    return -y_pred.log_prob(y_true)