"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """

    self.params = {}
    self.grads = {}

    self.params['weight'] = np.random.normal(0, 0.0001, in_features*out_features).reshape(out_features,in_features)
    self.params['bias'] = np.zeros((out_features,1))



  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    self.x = x
    out =  self.params['weight']  @ x.T+ self.params['bias']

    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """
    self.grads['weight'] = dout.T @ self.x

    # save grad towards b, with reshape to allow for different size bias for different layerplace of Linearmodule
    self.grads['bias'] = np.reshape(dout.T.sum(axis=1), self.params['bias'].shape)
    out =  dout @ self.params['weight']

    return out




class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    # from  https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # modified with axis=-1 and keepdims to be able to handle batch inputs, as the url is for single inputs
    def exp_normalize(self, x):
      b = x.max(axis=1,keepdims=True) 
      y = np.exp(x- b)
      return y/y.sum(axis=1, keepdims=True)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        out = self.exp_normalize(x)
        self.old_output = out
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        # with the help of einsum explenation of: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        # [:,None,:] used since the outer product (einsum) is 3D
        dx = dout * self.old_output - (dout[:,None,:] @  np.einsum('ki,kj->kij', self.old_output, self.old_output)).squeeze()
        
        return dx


class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    out =  np.sum(-y * np.log(x), axis=1) / y.shape[0]

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """
    dx = -y / x / y.shape[0]

    return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # x is needed for backwards pass
        self.old_input = x
        #np.nantonum is added since 0 * nan = nan and np.exp is evaluated for entire x
        out = (x >=0 ) * x + (x<0)* np.nan_to_num(np.exp(x)-1)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        #np.nantonum is added since 0 * nan = nan and np.exp is evaluated for entire x
        dx = dout*  ((self.old_input >=0 ) + (self.old_input<0)* np.nan_to_num(np.exp(self.old_input)))

        return dx
