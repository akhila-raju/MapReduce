import numpy as np
import cPickle as pickle
from time import time
from classifier import Classifier
from util.layers import *
from util.dump import *

""" STEP3: Build Deep Convolutional Neural Network """

class CNNClassifier(Classifier):
  def __init__(self, D, H, W, K, iternum):
    Classifier.__init__(self, D, H, W, K, iternum)

    """ 
    Layer 1 Parameters (Conv 32 x 32 x 16) 
    K = 16, F = 5, S = 1, P = 2
    weight matrix: [K1 * D * F1 * F1]
    bias: [K1 * 1]
    """
    K1, F1, self.S1, self.P1 = 16, 5, 1, 2
    self.A1 = 0.01 * np.random.randn(K1, D, F1, F1)
    self.b1 = np.zeros((K1, 1))
    H1 = (H - F1 + 2*self.P1) / self.S1 + 1
    W1 = (W - F1 + 2*self.P1) / self.S1 + 1

    """ 
    Layer 3 Parameters (Pool 16 x 16 x 16) 
    K = 16, F = 2, S = 2
    """
    K3, self.F3, self.S3 = K1, 2, 2
    H3 = (H1 - self.F3) / self.S3 + 1
    W3 = (W1 - self.F3) / self.S3 + 1
 
    """ 
    Layer 4 Parameters (Conv 16 x 16 x 20) 
    K = 20, F = 5, S = 1, P = 2
    weight matrix: [K4 * K3 * F4 * F4]
    bias: [K4 * 1]
    """
    K4, F4, self.S4, self.P4 = 20, 5, 1, 2
    self.A4 = 0.01 * np.random.randn(K4, K3, F4, F4)
    self.b4 = np.zeros((K4, 1))
    H4 = (H3 - F4 + 2*self.P4) / self.S4 + 1
    W4 = (W3 - F4 + 2*self.P4) / self.S4 + 1

    """ 
    Layer 6 Parameters (Pool 8 x 8 x 20) 
    K = 20, F = 2, S = 2
    """
    K6, self.F6, self.S6 = K4, 2, 2
    H6 = (H4 - self.F6) / self.S6 + 1
    W6 = (W4 - self.F6) / self.S6 + 1

    """ 
    Layer 7 Parameters (Conv 8 x 8 x 20) 
    K = 20, F = 5, S = 1, P = 2
    weight matrix: [K7 * K6 * F7 * F7]
    bias: [K7 * 1]
    """
    K7, F7, self.S7, self.P7 = 20, 5, 1, 2
    self.A7 = 0.01 * np.random.randn(K7, K6, F7, F7)
    self.b7 = np.zeros((K7, 1))
    H7 = (H6 - F7 + 2*self.P7) / self.S7 + 1
    W7 = (W6 - F7 + 2*self.P7) / self.S7 + 1

    """ 
    Layer 9 Parameters (Pool 4 x 4 x 20) 
    K = 20, F = 2, S = 2
    """
    K9, self.F9, self.S9 = K7, 2, 2
    H9 = (H7 - self.F9) / self.S9 + 1
    W9 = (W7 - self.F9) / self.S9 + 1

    """ 
    Layer 10 Parameters (FC 1 x 1 x K)
    weight matrix: [(K6 * H_6 * W_6) * K] 
    bias: [1 * K]
    """
    self.A10 = 0.01 * np.random.randn(K9 * H9 * W9, K)
    self.b10 = np.zeros((1, K))

    """ Hyperparams """
    # learning rate
    self.rho = 1e-2
    # momentum
    self.mu = 0.9
    # reg strength
    self.lam = 0.1
    # velocity for A1: [K1 * D * F1 * F1]
    self.v1 = np.zeros((K1, D, F1, F1))
    # velocity for A4: [K4 * K3 * F4 * F4]
    self.v4 = np.zeros((K4, K3, F4, F4))
    # velocity for A7: [K7 * K6 * F7 * F7]
    self.v7 = np.zeros((K7, K6, F7, F7))
    # velocity for A10: [(K9 * H9 * W9) * K]   
    self.v10 = np.zeros((K9 * H9 * W9, K))
 
    return

  def load(self, path):
    data = pickle.load(open(path + "layer1"))
    assert(self.A1.shape == data['w'].shape)
    assert(self.b1.shape == data['b'].shape)
    self.A1 = data['w']
    self.b1 = data['b'] 
    data = pickle.load(open(path + "layer4"))
    assert(self.A4.shape == data['w'].shape)
    assert(self.b4.shape == data['b'].shape)
    self.A4 = data['w']
    self.b4 = data['b']
    data = pickle.load(open(path + "layer7"))
    assert(self.A7.shape == data['w'].shape)
    assert(self.b7.shape == data['b'].shape)
    self.A7 = data['w']
    self.b7 = data['b']
    data = pickle.load(open(path + "layer10"))
    assert(self.A10.shape == data['w'].shape)
    assert(self.b10.shape == data['b'].shape)
    self.A10 = data['w']
    self.b10 = data['b']
    return 

  def param(self):
    return [
      ("A10", self.A10), ("b10", self.b10),
      ("A7", self.A7), ("b7", self.b7), 
      ("A4", self.A4), ("b4", self.b4), 
      ("A1", self.A1), ("b1", self.b1)] 

  def forward(self, data):
    """
    INPUT:
      - data: RDD[(key, (images, labels)) pairs]
    OUTPUT:
      - RDD[(key, (images, list of layers, labels)) pairs]
    """

    """ TODO: Layer1: Conv (32 x 32 x 16) forward """

    """ TODO: Layer2: ReLU (32 x 32 x 16) forward """

    """ DOTO: Layer3: Pool (16 x 16 x 16) forward """

    """ TODO: Layer4: Conv (16 x 16 x 20) forward """ 

    """ TODO: Layer5: ReLU (16 x 16 x 20) forward """

    """ TODO: Layer6: Pool (8 x 8 x 20) forward """ 

    """ TODO: Layer7: Conv (8 x 8 x 20) forward """ 

    """ TODO: Layer8: ReLU (8 x 8 x 20) forward """ 

    """ TODO: Layer9: Pool (4 x 4 x 20) forward """ 

    """ TODO: Layer10: FC (1 x 1 x 10) forward """
    A1 = self.A1
    b1 = self.b1
    A4 = self.A4
    b4 = self.b4
    A7 = self.A7
    b7 = self.b7
    S7 = self.S7
    P7 = self.P7
    A10 = self.A10
    b10 = self.b10
    S1 = self.S1
    P1 = self.P1
    F3 = self.F3
    S3 = self.S3
    S4 = self.S4
    P4 = self.P4
    F6 = self.F6
    S6 = self.S6
    F9 = self.F9
    S9 = self.S9

    # return data.map(lambda (k, (x, y)): (k, (x, [linear_forward(x, A1, b1), 
    #     ReLU_forward(linear_forward(x, A1, b1)), 
    #     linear_forward(ReLU_forward(linear_forward(x, A1, b1)), A3, b3)], y)))

    return data.map(lambda (k, (x, y)): (k, (x, [conv_forward(x, A1, b1, S1, P1), 
        ReLU_forward((conv_forward(x, A1, b1, S1, P1))[0]), 
        max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3),
        conv_forward(max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3)[0], A4, b4, S4, P4), 
        ReLU_forward(conv_forward(max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3)[0], A4, b4, S4, P4)[0]), 
        max_pool_forward((ReLU_forward(conv_forward(max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3)[0], A4, b4, S4, P4)[0])) ,F6, S6), 
        conv_forward((max_pool_forward((ReLU_forward(conv_forward(max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3)[0], A4, b4, S4, P4)[0])), F6, S6))[0], A7, b7, S7, P7),
        ReLU_forward(conv_forward((max_pool_forward((ReLU_forward(conv_forward(max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3)[0], A4, b4, S4, P4)[0])), F6, S6))[0], A7, b7, S7, P7)[0]),
        max_pool_forward(( ReLU_forward(conv_forward((max_pool_forward((ReLU_forward(conv_forward(max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3)[0], A4, b4, S4, P4)[0])), F6, S6))[0], A7, b7, S7, P7)[0])) , F9, S9),
        linear_forward((max_pool_forward(( ReLU_forward(conv_forward((max_pool_forward((ReLU_forward(conv_forward(max_pool_forward(ReLU_forward((conv_forward(x, A1, b1, S1, P1)[0])), F3, S3)[0], A4, b4, S4, P4)[0])), F6, S6))[0], A7, b7, S7, P7)[0])) , F9, S9)[0]), A10, b10)], y)))

  def backward(self, data, count):
    """
    INPUT:
      - data: RDD[(images, list of layers, labels) pairs]
    OUTPUT:
      - Loss
    """

    """ TODO: Softmax Loss Layer """ 
    softmax = data.map(lambda (x, l, y): (x, softmax_loss(l[-1], y) + (l,))) \
                  .map(lambda (x, (L, df, a)): (x, (L/count, df/count, a)))
    """ TODO: Compute Loss """
    L = softmax.map(lambda (x, (y, z, a)): y).reduce(lambda x, y: x + y)
    """ regularization """
    L += 0.5 * self.lam * np.sum(self.A1*self.A1)
    L += 0.5 * self.lam * np.sum(self.A4*self.A4)
    L += 0.5 * self.lam * np.sum(self.A7*self.A7)
    L += 0.5 * self.lam * np.sum(self.A10*self.A10)

    """ TODO: Layer10: FC (1 x 1 x 10) Backward """
    back_p_l10 = softmax.map(lambda (k, (x, y, a)): (k, linear_backward(y, a[8][0], self.A10) + (a,)))

    """ TODO: gradients on A10 & b10 """
    dLdA10 = back_p_l10.map(lambda (k, (x, y, z, a)): y).reduce(lambda x, y: y + x)
    dLdb10 = back_p_l10.map(lambda (k, (x, y, z, a)): z).reduce(lambda x, y: y + x)

    """ TODO: Layer9: Pool (4 x 4 x 20) Backward """
    back_p_l9 = back_p_l10.map(lambda (k, (x, y, z, a)): (k, (max_pool_backward(x, a[7], a[8][1], self.F9, self.S9), a)))
    """ TODO: Layer8: ReLU (8 x 8 x 20) Backward """
    back_p_l8 = back_p_l9.map(lambda (k, (x, a)): (k, (ReLU_backward(x, a[6][0]), a)))
    """ TODO: Layer7: Conv (8 x 8 x 20) Backward """
    back_p_l7 = back_p_l8.map(lambda (k, (x, a)): (k, conv_backward(x, a[5][0], a[6][1], self.A7, self.S7, self.P7) + (a,)))
    """ TODO: gradients on A7 & b7 """
    dLdA7 = back_p_l7.map(lambda (k, (x, y, z, a)): y).reduce(lambda x, y: y + x)
    dLdb7 = back_p_l7.map(lambda (k, (x, y, z, a)): z).reduce(lambda x, y: y + x)
 
    """ TODO: Layer6: Pool (8 x 8 x 20) Backward """
    back_p_l6 = back_p_l7.map(lambda (k, (x, y, z, a)): (k, (max_pool_backward(x, a[4], a[5][1], self.F6, self.S6), a)))
    """ TODO: Layer5: ReLU (16 x 16 x 20) Backward """ 
    back_p_l5 = back_p_l6.map(lambda (k, (x, a)): (k, (ReLU_backward(x, a[3][0]), a)))
    """ TODO: Layer4: Conv (16 x 16 x 20) Backward """ 
    back_p_l4 = back_p_l5.map(lambda (k, (x, a)): (k, conv_backward(x, a[2][0], a[3][1], self.A4, self.S4, self.P4) + (a,)))
    """ TODO: gradients on A4 & b4 """
    dLdA4 = back_p_l4.map(lambda (k, (x, y, z, a)): y).reduce(lambda x, y: y + x)
    dLdb4 = back_p_l4.map(lambda (k, (x, y, z, a)): z).reduce(lambda x, y: y + x)
 
    """ TODO: Layer3: Pool (16 x 16 x 16) Backward """ 
    back_p_l3 = back_p_l4.map(lambda (k, (x, y, z, a)): (k, (max_pool_backward(x, a[1], a[2][1], self.F3, self.S3), a)))
    """ TODO: Layer2: ReLU (32 x 32 x 16) Backward """
    back_p_l2 = back_p_l3.map(lambda (k, (x, a)): (k, (ReLU_backward(x, a[0][0]), a)))
    """ TODO: Layer1: Conv (32 x 32 x 16) Backward """
    back_p_l1 = back_p_l2.map(lambda (k, (x, a)): (k, conv_backward(x, k, a[0][1], self.A1, self.S1, self.P1) + (a,)))
    """ TODO: gradients on A1 & b1 """
    dLdA1 = back_p_l1.map(lambda (k, (x, y, z, a)): y).reduce(lambda x, y: y + x)
    dLdb1 = back_p_l1.map(lambda (k, (x, y, z, a)): z).reduce(lambda x, y: y + x)
 

    """ regularization gradient """
    dLdA10 = dLdA10.reshape(self.A10.shape)
    dLdA7 = dLdA7.reshape(self.A7.shape)
    dLdA4 = dLdA4.reshape(self.A4.shape)
    dLdA1 = dLdA1.reshape(self.A1.shape)
    dLdA10 += self.lam * self.A10
    dLdA7 += self.lam * self.A7
    dLdA4 += self.lam * self.A4
    dLdA1 += self.lam * self.A1

    """ tune the parameter """
    self.v1 = self.mu * self.v1 - self.rho * dLdA1
    self.v4 = self.mu * self.v4 - self.rho * dLdA4
    self.v7 = self.mu * self.v7 - self.rho * dLdA7
    self.v10 = self.mu * self.v10 - self.rho * dLdA10
    self.A1 += self.v1
    self.A4 += self.v4 
    self.A7 += self.v7
    self.A10 += self.v10
    self.b1 += -self.rho * dLdb1
    self.b4 += -self.rho * dLdb4
    self.b7 += -self.rho * dLdb7
    self.b10 += -self.rho * dLdb10

    return L

