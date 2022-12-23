import numpy as np

class DigitPredictor:
    '''
    The tensorflow library was taking up too much space on Heroku, so I stored the model weights in an npz file and created this
    class to make the predictions using only numpy.

    Attributes:
        middle_layers:
            a numpy array containing the weights of each dense layer in the keras model
        output_layer:
            the weights of the softmax layer of the keras model
    '''

    def __init__(self,npz_file):
        self.kernel_1 = npz_file.get('kernel_1')
        self.kernel_2 = npz_file.get('kernel_2')
        self.output_kernel = npz_file.get('output_kernel')

    def relu(self,x):
        return np.maximum(0,x)

    def predict(self,input):
        input = input.flatten()
        input = self.relu(input@self.kernel_1)
        input= self.relu(input@self.kernel_2)
        output = np.argmax(input@self.output_kernel)
        return output