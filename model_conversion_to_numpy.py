import numpy as np
from tensorflow import keras   

model = keras.models.load_model("my_first.model")
kernel_1 = model.get_layer('dense').get_weights()[0]
kernel_2 = model.get_layer('dense_1').get_weights()[0]
output_kernel = model.get_layer('dense_2').get_weights()[0]

np.savez('model_weights.npz',kernel_1=kernel_1,kernel_2=kernel_2,output_kernel=output_kernel)


test = np.load('model_weights.npz')
print(test.get('kernel_1'))