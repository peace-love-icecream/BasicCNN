from Layers.Base import BaseLayer
from scipy.signal import correlate
from scipy.signal import convolve
import numpy as np
from Layers.Initializers import *

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self._optimizerWeights = None
        self._optimizerBias = None
        self.weights = np.ones(((self.num_kernels,) + self.convolution_shape))
        self.bias = np.ones((self.num_kernels, 1))
        self.initialize(UniformRandom(), UniformRandom())
        self._gradient_weights = None
        self._gradient_bias = None
        self.input = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizerWeights(self):
        return self._optimizerWeights

    @property
    def optimizerBias(self):
        return self._optimizerBias

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @optimizerWeights.setter
    def optimizer(self, value):
        self._optimizerWeights = value

    @optimizerBias.setter
    def optimizer(self, value):
        self._optimizerBias = value

    def forward(self, input_tensor):
        self.input = input_tensor
        if len(input_tensor.shape) == 4:
            batch_size, input_channel, input_x, input_y = input_tensor.shape
            ##print(self.stride_shape)
            ##print("input_tensor ", input_tensor.shape)
            ##print("kernel ", self.convolution_shape)
            ##print("weights ", self.weights.shape)
            ##print("bias ", self.bias.shape)
            output_x = int(np.ceil(input_x / self.stride_shape[0]))
            output_y = int(np.ceil(input_y / self.stride_shape[1]))
            output = np.zeros((batch_size, self.num_kernels, output_x, output_y))
            ##print("output_tensor ", output.shape)
            for i, batch in enumerate(input_tensor):
                temp_output = np.zeros((self.num_kernels, output_x, output_y))
                # pad input
                pad_xL = int(np.floor(self.convolution_shape[1] / 2))
                pad_xR = int(np.ceil(self.convolution_shape[1] / 2 - 1))
                pad_yT = int(np.floor(self.convolution_shape[2] / 2))
                pad_yB = int(np.ceil(self.convolution_shape[2] / 2 - 1))
                input_pad = np.pad(batch, pad_width=((0,0), (pad_xL, pad_xR), (pad_yT, pad_yB)))
                ##print("input_pad ", input_pad.shape)

                for j in range(self.num_kernels):
                    ##print("batch ", batch.shape)
                    ##print("weights ", self.weights[j].shape)
                    ##print("correlation ", correlate(batch, self.weights[j], mode='same').shape)
                    cor = correlate(input_pad, self.weights[j], mode='valid')
                    # middle channel
                    #cor = cor[int(np.floor(input_channel / 2))]
                    # stride
                    firstx = 0
                    lastx = (int(np.ceil(input_x / self.stride_shape[0])) - 1) * self.stride_shape[0]
                    firsty = 0
                    lasty = (int(np.ceil(input_y / self.stride_shape[1])) - 1) * self.stride_shape[1]
                    ##print("lastx ", lastx)
                    ##print("lasty ", lasty)
                    ##print("correlation ", cor.shape)
                    cor = cor[0, firstx:lastx + 1:self.stride_shape[0], firsty:lasty + 1:self.stride_shape[1]]
                    # elementwise kernelwise bias addition
                    ##print("correlation after stride ", cor.shape)
                    cor = cor + self.bias[j]
                    temp_output[j] = cor
                output[i] = temp_output
        else:
            batch_size, input_channel, input_x = input_tensor.shape
            ##print("input_tensor ", input_tensor.shape)
            ##print("kernel ", self.convolution_shape)
            ##print("weights ", self.weights.shape)
            ##print("bias ", self.bias.shape)
            output_x = int(np.ceil(input_x / self.stride_shape[0]))
            output = np.zeros((batch_size, self.num_kernels, output_x))
            ##print("output_tensor ", output.shape)
            for i, batch in enumerate(input_tensor):
                temp_output = np.zeros((self.num_kernels, output_x))
                # pad input
                pad_xL = int(np.floor(self.convolution_shape[1] / 2))
                pad_xR = int(np.ceil(self.convolution_shape[1] / 2 - 1))
                input_pad = np.pad(batch, pad_width=((0, 0), (pad_xL, pad_xR)))
                for j in range(self.num_kernels):
                    ##print("batch ", batch.shape)
                    ##print("weights ", self.weights[j].shape)
                    ##print("correlation ", correlate(batch, self.weights[j], mode='same').shape)
                    cor = correlate(input_pad, self.weights[j], mode='valid')
                    # middle channel
                    #cor = cor[int(np.floor(input_channel / 2))]
                    # stride
                    firstx = 0
                    lastx = (int(np.ceil(input_x / self.stride_shape[0])) - 1) * self.stride_shape[0]
                    ##print("lastx ", lastx)
                    ##print(cor.shape)
                    cor = cor[0, firstx:lastx + 1:self.stride_shape[0]]
                    ##print("correlation after stride ", cor.shape)
                    # elementwise kernelwise bias addition
                    cor = cor + self.bias[j]
                    temp_output[j] = cor
                output[i] = temp_output
        return output

    def backward(self, error_tensor):
        if len(self.input.shape) == 4:

            _, input_channel, inputX, inputY = self.input.shape
            batch, output_channel, outputX, outputY = error_tensor.shape
            input_channel, kernel_x, kernel_y = self.convolution_shape

            error_tensor_upscaled = np.zeros((batch, output_channel, inputX, inputY))
            new_kernel = np.zeros((input_channel, self.num_kernels, kernel_x, kernel_y))
            output_error = np.zeros(self.input.shape)

            for i, error in enumerate(error_tensor):
                print("Batch ", i)
                print("ErrorTensor ", error.shape)
                # upscale error tensor (fill in error_tensor in with zero initialized array)
                for j in range(0, self.num_kernels):
                    error_tensor_upscaled[i, j][::self.stride_shape[0], ::self.stride_shape[1]] = error[j]
                print("Stride padded error_tensor ", error_tensor_upscaled.shape)
                # pad upscaled error tensor
                pad_xL = int(np.floor(self.convolution_shape[1] / 2))
                pad_xR = int(np.ceil(self.convolution_shape[1] / 2 - 1))
                pad_yT = int(np.floor(self.convolution_shape[2] / 2))
                pad_yB = int(np.ceil(self.convolution_shape[2] / 2 - 1))
                error_tensor_upscaled_pad = np.pad(error_tensor_upscaled[i], pad_width=((0, 0), (pad_xL, pad_xR), (pad_yT, pad_yB)))
                print("error_pad (stride + convolve padding: ", error_tensor_upscaled_pad.shape)

                #for j in range(0, self.)
                #gradient with respect to bias
                #self.gradient_bias = np.sum(error_tensor[i], axis=1).sum(axis=1)
                #print("gradient bias ", self.gradient_bias.shape)

                #gradient with respect to weights
                print("GRADIENT WITH RESPECT TO WEIGHTS:")
                #print("batch ", batch.shape)
                #print("input ", self.input[i].shape)
                # pad input
                pad_xL = int(np.floor(self.convolution_shape[1] / 2))
                pad_xR = int(np.ceil(self.convolution_shape[1] / 2 - 1))
                pad_yT = int(np.floor(self.convolution_shape[2] / 2))
                pad_yB = int(np.ceil(self.convolution_shape[2] / 2 - 1))
                input_pad = np.pad(self.input[i], pad_width=((0, 0), (pad_xL, pad_xR), (pad_yT, pad_yB)))
                print("input_pad ", input_pad.shape)
                # upscale error tensor (fill in error_tensor in with zero initialized array)
                for j in range(0, self.num_kernels):
                    error_tensor_upscaled[i, j][::self.stride_shape[0], ::self.stride_shape[1]] = error[j]
                print("error ", error_tensor_upscaled.shape)
                grad_weights = np.ones((self.convolution_shape))
                # correlate input with error tensor
                cor = correlate(error_tensor_upscaled, input_pad, mode='valid')
                grad_weights[j] = cor
                self.gradient_weights = grad_weights
                print("gradient weights ", self.gradient_weights.shape)

                #gradient with respect to lower layer
                print("GRADIENT WITH RESPECT TO LOWER LAYER:")

                # flip kernel -> reorder kernel
                for j in range(0, input_channel):
                    for k in range(0, self.num_kernels):
                        new_kernel[j, k] = self.weights[k, j]




                #print(error_pad)


                # loop over new kernels (#input_channel)
                for j in range(0, input_channel):
                    print("kernel ", new_kernel[j].shape)
                    conv = convolve(error_pad, new_kernel[j], mode='valid')
                    print("conv ", conv[0].shape)
                    output_error[i, j] = conv[0] * (-1)

                print("error_tensorOut ", output_error.shape)

                if self.optimizerWeights:
                    pass
                    # update weights
                if self.optimizerBias:
                    pass
                    # update bias
        else:
            for i, batch in enumerate(error_tensor):
                # for j in range(0, self.)
                # gradient with respect to bias
                self.gradient_bias = np.sum(error_tensor[i], axis=1).sum(axis=1)
                print("gradient bias ", self.gradient_bias.shape)

                # gradient with respect to weights
                # print("GRADIENT WITH RESPECT TO WEIGHTS:")
                # print("batch ", batch.shape)
                # print("input ", self.input[i].shape)
                # pad input
                # pad_xL = int(np.floor(self.convolution_shape[1] / 2))
                # pad_xR = int(np.ceil(self.convolution_shape[1] / 2 - 1))
                # pad_yT = int(np.floor(self.convolution_shape[2] / 2))
                # pad_yB = int(np.ceil(self.convolution_shape[2] / 2 - 1))
                # input_pad = np.pad(self.input[i], pad_width=((0, 0), (pad_xL, pad_xR), (pad_yT, pad_yB)))
                # print("input_pad ", input_pad.shape)
                # grad_weights = np.ones((self.convolution_shape))
                # correlate input with error tensor
                # for j in range(0, self.num_kernels):
                #    cor = correlate(batch[j], input_pad[j], mode='valid')
                #    grad_weights[j] = cor
                # self.gradient_weights = grad_weights
                # print("gradient weights ", self.gradient_weights.shape)

                # gradient with respect to lower layer
                print("GRADIENT WITH RESPECT TO LOWER LAYER:")
                print("input ", self.input.shape)
                print("error_tensor for batch ", i, ": ", batch.shape)
                print("old kernel ", self.weights.shape)
                # new kernel aka. kernel flip - loop inefficient recheck
                input_channel, kernel_x = self.convolution_shape
                kernel = np.ones((input_channel, self.num_kernels, kernel_x))
                for j in range(0, input_channel):
                    for k in range(0, self.num_kernels):
                        kernel[j, k] = self.weights[k, j]
                print("new kernel ", kernel.shape)

                # pad stride - has to be done for each channel
                zero_col = np.zeros((self.stride_shape[0] - 1, 1))
                # print("Zero Column ", zero_col.shape)
                # print("Zero Row ", zero_row.shape)
                # error_pad shape : outputchannel, inputx, inputy
                error_pad = np.zeros((kernel.shape[1], self.input[i].shape[1]))
                for k, spatial in enumerate(batch):
                    pad_stride = batch[i]
                    for j in np.arange(1, self.input[i].shape[1], self.stride_shape[0]):
                        if zero_col.shape[1] > 0:
                            # print("Col ", j)
                            pad_stride = np.insert(pad_stride, j, zero_col, axis=0)
                            # print(pad_stride.shape)
                    error_pad[k] = pad_stride
                print("Stride padded error_tensor ", error_pad.shape)

                # pad convolve
                pad_xL = int(np.floor(self.convolution_shape[1] / 2))
                pad_xR = int(np.ceil(self.convolution_shape[1] / 2 - 1))

                error_pad = np.pad(error_pad, pad_width=((0, 0), (pad_xL, pad_xR)))

                print("error_pad (stride + convolve padding: ", error_pad.shape)

                for j in range(0, input_channel):
                    print("gradient wrt to layer ", j)
                    print("conv kernel ", kernel[j].shape)
                    conv = convolve(error_pad, kernel[j], mode='valid')
                    print("conv ", conv.shape)
                    output_error[i, j] = conv
                print("error_tensorOut ", output_error.shape)

                if self.optimizerWeights:
                    pass
                    # update weights
                if self.optimizerBias:
                    pass
                    # update bias

        return output_error

    def initialize(self, weights_initializer, bias_initializer):
        if len(self.convolution_shape) == 3:
            input_channel, kernel_height, kernel_width = self.convolution_shape
            fan_in = np.prod(self.convolution_shape)
            fan_out = np.prod((self.num_kernels, kernel_height, kernel_width))
            self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
            self.bias = bias_initializer.initialize(self.bias.shape, None, None)
        else:
            input_channel, kernel = self.convolution_shape
            fan_in = np.prod(self.convolution_shape)
            fan_out = np.prod((self.num_kernels, kernel))
            self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
            self.bias = bias_initializer.initialize(self.bias.shape, None, None)

