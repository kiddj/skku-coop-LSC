import tensorflow as tf
import numpy as np

from AE_credit import encode_creidt
from data_credit import load_credit
from plot_credit import plot_decision_score

AE_modelPath = "models/ae_credit_16_notbest.h5"

# g = lambda x : relu(x)                # RELU
g = lambda x: 1 / (1 + tf.exp(-x))    # Sigmoid
# g = lambda x : x                      # Linear

def tf_OneClass_NN_Sigmoid(data_normal, data_anomaly):
    #tf.result_default_graph()

    X_normal = data_normal

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    #Layer Size
    x_size = X_normal.shape[1]  # input nodes
    h_size = 4                 # hidden nodes
    y_size = 1                  # outputs
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0, 1, (len(X_normal), y_size))
    nu = 0.04

    ''' Weight initialization '''
    def init_weights(shape):
        weights = tf.random_normal(shape, mean=0, stddev=0.00001)
        return tf.Variable(weights)

    ''' Forward Propagation '''
    def forwardprop(X, w_1, w_2):
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h = tf.nn.sigmoid(tf.matmul(X, w_1))
        yhat = tf.matmul(h, w_2)
        return yhat

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g, r):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)

        term1 = 0.5 * tf.reduce_sum(w**2)
        term2 = 0.5 * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))
        term4 = -r

        return term1 + term2 + term3 + term4

    X_anomaly = data_anomaly

    ''' Symbols '''
    X = tf.placeholder("float32", shape=[None, x_size])
    r = tf.get_variable("r", dtype=tf.float32, shape=(), trainable=False)

    # Weight init
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    cost = ocnn_obj(theta, X, nu, w_1, w_2, g, r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    #SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    epochs = 400

    for epoch in range(epochs):
        sess.run(updates, feed_dict={X: X_normal, r:rvalue})
        rvalue = nnScore(X_normal, w_1, w_2, g)
        with sess.as_default():
            rvalue = rvalue.eval()
            rvalue = np.percentile(rvalue, q=100*0.04)
        print("Epoch = %d, r = %f" % (epoch+1, rvalue))

    train = nnScore(X_normal, w_1, w_2, g)
    test = nnScore(X_anomaly, w_1, w_2, g)
    with sess.as_default():
        arrNormal = train.eval()
        arrAnomaly = test.eval()

    rstar = rvalue
    sess.close()

    print("Session Closed!")

    pos_decisionScore = arrNormal - rstar
    neg_decisionScore = arrAnomaly - rstar

    print("pos decisionScore\n", pos_decisionScore)
    print("neg decisionScore\n", neg_decisionScore)

    # Evaluate

    threshold = 1e-02

    print("Normal Data : ", len(pos_decisionScore))
    print("Normal - Detect Anomaly : ", (pos_decisionScore < threshold).sum())
    print("Anomaly Data : ", len(neg_decisionScore))
    print("Anomaly - Detect Anomaly : ", (neg_decisionScore < threshold).sum())

    return [pos_decisionScore, neg_decisionScore]

[credit_normal, credit_anomaly] = load_credit()

[credit_normal_ae, credit_anomaly_ae] = \
    encode_creidt(credit_normal, credit_anomaly, AE_modelPath)

print("Normal AE", credit_normal_ae)
print("Anomaly AE", credit_anomaly_ae)

[pos_ds, neg_ds] = tf_OneClass_NN_Sigmoid(credit_normal_ae, credit_anomaly_ae)



# plot_decision_score(pos_ds, neg_ds)