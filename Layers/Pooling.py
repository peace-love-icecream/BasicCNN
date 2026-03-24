from Layers.Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.cache = {}

    def forward(self, input_tensor):
        #print("input_tensor.shape", input_tensor.shape)

        self.input_shape = input_tensor.shape
        batch_size, num_channel, input_x, input_y = input_tensor.shape
        x_out = 1 + (input_x - self.pooling_shape[0]) // self.stride_shape[0]
        y_out = 1 + (input_y - self.pooling_shape[1]) // self.stride_shape[1]
        output = np.zeros((batch_size, num_channel, x_out, y_out))

        #input_x = 3
        #print("input_x =", input_x)
        #print("pooling_shape[0] =", self.pooling_shape[0])
        #print("input_x - self.pooling_shape[0] =", input_x - self.pooling_shape[0])
        #print("stride_shape[0] =", self.stride_shape[0])
        #print("(input_x - self.pooling_shape[0]) // self.stride_shape[0] =", (input_x - self.pooling_shape[0]) // self.stride_shape[0])
        #print("output_x", output_x)

        #print("stride_shape", self.stride_shape)
        #print("pooling_shape", self.pooling_shape)
        #print("output.shape", output.shape)

        for x in range(x_out):
            for y in range(y_out):
                x_start = x * self.stride_shape[0]
                x_end = x_start + self.pooling_shape[0]
                y_start = y * self.stride_shape[1]
                y_end = y_start + self.pooling_shape[1]
                slice = input_tensor[:, :, x_start:x_end, y_start:y_end]
                self.save_mask(slice, (x, y))
                output[:, :, x, y] = np.max(slice, axis=(2, 3))

        return output

    def backward(self, error_tensor):
        output = np.zeros(self.input_shape)
        _, _, x_out, y_out = error_tensor.shape

        for x in range(x_out):
            for y in range(y_out):
                x_start = x * self.stride_shape[0]
                x_end = x_start + self.pooling_shape[0]
                y_start = y * self.stride_shape[1]
                y_end = y_start + self.pooling_shape[1]
                output[:, :, x_start:x_end, y_start:y_end] += error_tensor[:, :, x:x + 1, y:y + 1] * self.cache[(x, y)]

        return output

    def save_mask(self, slice, coordinates):
        mask = np.zeros(slice.shape)
        b, n, x, y = slice.shape
        arr = slice.reshape(b, n, x * y)
        idx = np.argmax(arr, axis=2)

        b_idx, n_idx = np.indices((b, n))
        mask.reshape(b, n, x * y)[b_idx, n_idx, idx] = 1
        self.cache[coordinates] = mask
