import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tools.common_tools import setup_seed


class Floor(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input


class Round(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the round function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input


class Clamp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the clamp function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None, None


class QuantConv2d(nn.Conv2d):
    """
    UINT Quantized Conv2d Module.
        in_channels: input channel
        out_channels: output channel
        kernel_size: kernel size
        stride: stride
        padding: padding
        dilation: dilation
        groups: groups
        bias: bias
        padding_mode: padding mode
        wbit: bit width of weight
        xbit: bit width of input activation
        mode: operation mode, e.g.: 'Train' or 'Inference'
        device: 'cpu' or 'cuda' or 'mps' depend on your device
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 wbit=8,
                 xbit=8,
                 mode='Train',
                 device='cpu'):
        super(QuantConv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          groups,
                                          bias,
                                          padding_mode)
        assert mode in ['Train', 'Inference'], "Invalid mode specified. Choose from 'Train', 'Inference'."
        self.wbit = wbit
        self.xbit = xbit
        self.kernel_size_param = kernel_size
        self.in_channels_param = in_channels
        self.padding_param = padding
        self.stride_param = stride
        self.bias_param = bias
        self.epsilon = 1e-7
        self.mode = mode
        self.device = device

    def _quantize_weight_train(self, input):
        """
        Quantize weight tensor in 'Train' mode.
        input: input weight tensor
        return: fake quantized weight tensor for training
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        w_bias = torch.min(input)
        input = torch.sub(input, w_bias)
        w_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / w_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.wbit - 1.0)) / (2.0 ** self.wbit - 1.0) # UINT Quantization
        return input * w_scale + w_bias

    def _quantize_feature_train(self, input):
        """
        Quantize input activation tensor in 'Train' mode.
        input: input activation tensor
        return: fake quantized input activation tensor for training
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_bias = torch.min(input)
        input = torch.sub(input, x_bias)
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0)) / (2.0 ** self.xbit - 1.0) # UINT Quantization
        return input * x_scale + x_bias

    def _quantize_weight_infer(self, input):
        """
        Quantize weight tensor in 'Inference' mode.
        input: input weight tensor
        return: quantized weight with UINT format for inference, scale & bias factor of weight, filter weight summation for dequantization
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        w_bias = torch.min(input)
        input = torch.sub(input, w_bias)
        w_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / w_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.wbit - 1.0)) # Quantization with UINT format
        # Calculate filter weight summation for dequantization
        fake_quant = torch.mul(input / (2.0 ** self.wbit - 1.0), w_scale)
        fake_quant = torch.add(fake_quant, w_bias)
        w_sum = torch.sum(fake_quant, dim=(1, 2, 3))
        return input, w_scale, w_bias, w_sum

    def _quantize_feature_infer(self, input):
        """
        Quantize input activation tensor in 'Inference' mode.
        input: input activation tensor
        return: quantized input activation tensor with UINT format for inference, scale & bias factor of input activation, filter activation summation for dequantization, UINT zero value
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_bias = torch.min(input)
        input = torch.sub(input, x_bias)
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0)) # Quantization with UINT format
        # Calculate zero point in UINT format using in padding
        x_zero = float(Round.apply(Clamp.apply(torch.abs(x_bias) / x_scale, 0.0, 1.0) * (2.0 ** self.xbit - 1.0)))
        # Calculate filter activation summation for dequantization
        batch, channel, height, width = input.shape
        fake_quant = torch.mul(input / (2.0 ** self.xbit - 1.0), x_scale)
        fake_quant = torch.add(fake_quant, x_bias)
        # Padding zero
        sum_map = F.pad(fake_quant,
                        (self.padding_param, self.padding_param, self.padding_param, self.padding_param),
                        'constant',
                        x_zero)
        height_o = math.floor((height - self.kernel_size_param + 2 * self.padding_param) / self.stride_param + 1)
        width_o = math.floor((width - self.kernel_size_param + 2 * self.padding_param) / self.stride_param + 1)
        # Initializing activation summation
        x_sum = torch.zeros(batch, height_o, width_o, device=self.device)
        # Kernel striding
        for i in range(0, height - self.kernel_size_param + 2 * self.padding_param + 1, self.stride_param):
            for j in range(0, width - self.kernel_size_param + 2 * self.padding_param + 1, self.stride_param):
                field = sum_map[:, :, i:i + self.kernel_size_param, j:j + self.kernel_size_param]
                x_sum[:, int(i / self.stride_param), int(j / self.stride_param)] = torch.sum(field, dim=(1, 2, 3))
        return input, x_scale, x_bias, x_sum, x_zero

    def forward(self, input):
        """
        Forward call of QuantConv2d with selective operation mode.
        input: input activation tensor
        return: output activation tensor
        """
        if self.mode == 'Train':    # Training mode
            w_quant = self._quantize_weight_train(self.weight)
            x_quant = self._quantize_feature_train(input)
            output = F.conv2d(x_quant,
                              w_quant,
                              self.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)
            output = output

        if self.mode == 'Inference':    # Inference mode that mimic UINT output
            w_quant, w_scale, w_bias, w_sum = self._quantize_weight_infer(self.weight)
            x_quant, x_scale, x_bias, x_sum = self._quantize_feature_infer(input)[0:4]
            output = F.conv2d(x_quant,
                              w_quant,
                              bias=self.bias,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups)
            # De-quantization to FP32
            batch, channel, height, width = output.shape
            w_sum = w_sum.repeat(batch, height, width, 1).permute(0, 3, 1, 2)
            x_sum = x_sum.repeat(channel, 1, 1, 1).permute(1, 0, 2, 3)
            output = output * (w_scale * x_scale) / ((2.0 ** self.wbit - 1.0) * (2.0 ** self.xbit - 1.0))
            output = output + x_bias * w_sum + w_bias * x_sum - w_bias * x_bias * self.in_channels_param * self.kernel_size_param * self.kernel_size_param
            # Compensation bias if needed
            if self.bias_param:
                x_height = output.shape[-2]
                x_width = output.shape[-1]
                bias_infer = self.bias.repeat(x_height, x_width, 1).permute(2, 0, 1)
                output += bias_infer

        return output


class PAConv2d(nn.Conv2d):
    """
    Bit-wise simulation PAConv2d Module.
        in_channels: input channel
        out_channels: output channel
        kernel_size: kernel size
        stride: stride
        padding: padding
        dilation: dilation
        groups: groups
        bias: bias
        padding_mode: padding mode
        wbit: bit width of weight
        xbit: bit width of input activation
        operand: PAC operand
        dynamic_config: enable the dynamic cycle configuration -> use sparsity to speculate the output activation
        threshold: threshold (th0, th1, th2) for cycle configuration (10/13/15/16), e.g.: (7000.0, 10000.0, 13000.0)
        mode: operation mode, e.g.: 'Train' or 'Inference' or 'Simulation'
        trim_noise: standard deviation applied for noise-aware training in 'Train' mode or roughly evaluate noise impact in 'Inference' mode
        device: 'cpu' or 'cuda' or 'mps' depend on your device
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 wbit=8,
                 xbit=8,
                 operand=4,
                 dynamic_config=False,
                 threshold=(7000.0, 10000.0, 13000.0),
                 mode='Train',
                 trim_noise=0.0,
                 device='cpu'):
        super(PAConv2d, self).__init__(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dilation,
                                       groups,
                                       bias,
                                       padding_mode)
        assert mode in ['Train', 'Inference', 'Simulation'], "Invalid mode specified. Choose from 'Train', 'Inference', or 'Simulation'."
        if dynamic_config:
            assert isinstance(threshold, tuple) and len(threshold) == 3, "Invalid threshold. Give threshold (TH0, TH1, TH2)."
            assert (operand == 4) and (wbit == 8) and (xbit == 8), "Dynamic configuration is implemented in 4-bit operand PAC in 8-bit task."
        self.wbit = wbit
        self.xbit = xbit
        self.operand = operand
        self.dynamic_config = dynamic_config
        self.threshold = threshold
        self.kernel_size_param = kernel_size
        self.in_channels_param = in_channels
        self.padding_param = padding
        self.stride_param = stride
        self.bias_param = bias
        self.epsilon = 1e-7
        self.mode = mode
        self.trim_noise = trim_noise / 100
        self.device = device

    def _quantize_weight_train(self, input):
        """
        Quantize weight tensor in 'Train' mode.
        input: input weight tensor
        return: fake quantized weight tensor for training
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        w_bias = torch.min(input)
        input = torch.sub(input, w_bias)
        w_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / w_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.wbit - 1.0)) / (2.0 ** self.wbit - 1.0) # UINT Quantization
        return input * w_scale + w_bias

    def _quantize_feature_train(self, input):
        """
        Quantize input activation tensor in 'Train' mode.
        input: input activation tensor
        return: fake quantized input activation tensor for training
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_bias = torch.min(input)
        input = torch.sub(input, x_bias)
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0)) / (2.0 ** self.xbit - 1.0) # UINT Quantization
        return input * x_scale + x_bias

    def _quantize_weight_infer(self, input):
        """
        Quantize weight tensor in 'Inference' mode.
        input: input weight tensor
        return: quantized weight with UINT format for inference, scale & bias factor of weight, filter weight summation for dequantization
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        w_bias = torch.min(input)
        input = torch.sub(input, w_bias)
        w_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / w_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.wbit - 1.0)) # Quantization with UINT format
        # Calculate filter weight summation for dequantization
        fake_quant = torch.mul(input / (2.0 ** self.wbit - 1.0), w_scale)
        fake_quant = torch.add(fake_quant, w_bias)
        w_sum = torch.sum(fake_quant, dim=(1, 2, 3))
        return input, w_scale, w_bias, w_sum

    def _quantize_feature_infer(self, input):
        """
        Quantize input activation tensor in 'Inference' mode.
        input: input activation tensor
        return: quantized input activation tensor with UINT format for inference, scale & bias factor of input activation, filter activation summation for dequantization, UINT zero value
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_bias = torch.min(input)
        input = torch.sub(input, x_bias)
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0)) # Quantization with UINT format
        # Calculate zero point in UINT format using in padding
        x_zero = float(Round.apply(Clamp.apply(torch.abs(x_bias) / x_scale, 0.0, 1.0) * (2.0 ** self.xbit - 1.0)))
        # Calculate filter activation summation for dequantization
        batch, channel, height, width = input.shape
        fake_quant = torch.mul(input / (2.0 ** self.xbit - 1.0), x_scale)
        fake_quant = torch.add(fake_quant, x_bias)
        # Padding zero
        sum_map = F.pad(fake_quant,
                        (self.padding_param, self.padding_param, self.padding_param, self.padding_param),
                        'constant',
                        x_zero)
        height_o = math.floor((height - self.kernel_size_param + 2 * self.padding_param) / self.stride_param + 1)
        width_o = math.floor((width - self.kernel_size_param + 2 * self.padding_param) / self.stride_param + 1)
        # Initializing activation summation
        x_sum = torch.zeros(batch, height_o, width_o, device=self.device)
        # Kernel striding
        for i in range(0, height - self.kernel_size_param + 2 * self.padding_param + 1, self.stride_param):
            for j in range(0, width - self.kernel_size_param + 2 * self.padding_param + 1, self.stride_param):
                field = sum_map[:, :, i:i + self.kernel_size_param, j:j + self.kernel_size_param]
                x_sum[:, int(i / self.stride_param), int(j / self.stride_param)] = torch.sum(field, dim=(1, 2, 3))
        return input, x_scale, x_bias, x_sum, x_zero

    def _decompose_weight(self, input):
        """
        Decompose FP32 weight into UINT quantized tensor with '0' and '1'.
        input: input weight tensor in FP32
        return: decomposed UINT result tensor with '0' and '1' in shape (wbit, filter_num, channel, height, width)
        """
        filter_num, channel, height, width = input.shape
        # Create tensor to store decomposed UINT bit results
        w_map = torch.tensor([], device=self.device)
        # Loop to integrate remaining bits
        for i in range(self.wbit - 1, -1, -1):
            w_map = torch.cat((w_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
            input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (wbit, filter_num, channel, height, width), e.g.: (8, 128, 64, 32, 32)
        w_map = w_map.reshape(self.wbit, filter_num, channel, height, width)
        return w_map

    def _decompose_feature(self, input):
        """
        Decompose FP32 input activation into UINT quantized tensor with '0' and '1'.
        input: input activation tensor in FP32
        return: decomposed UINT result tensor with '0' and '1' in shape (xbit, batch, channel, height, width)
        """
        batch, channel, height, width = input.shape
        # Create tensor to store decomposed UINT bit results
        x_map = torch.tensor([], device=self.device)
        # Loop to integrate remaining bits
        for i in range(self.xbit-1, -1, -1):
            x_map = torch.cat((x_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
            input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (xbit, batch, channel, height, width), e.g.: (8, 4, 64, 32, 32)
        x_map = x_map.reshape(self.xbit, batch, channel, height, width)
        return x_map

    def _speculate_output(self, input, kernel, channel):
        """
        Speculate the output activation based on bit-level sparsity for dynamic cycle configuration.
        E(O) = S(x7) * 2**7 + S(x6) * 2**6 + S(x5) * 2**5 + S(4) * 2**4 + S(3) * 2**3 + S(2) * 2**2 + S(1) * 2**1 + S(0) * 2**0 * 64 * 3 * 3 / (Kernel * Kernel * Channel)
        input: input UINT bit map
        kernel: kernel size
        channel: channel depth
        return: Output activation estimation based on bit sparsity
        """
        assert input.shape[1] == 1, "Do not support multi batch simulation in dynamic configuration."
        # Obtain each bit sparsity across kernel
        sparsity_map = torch.sum(input, (2, 3, 4)).squeeze()
        # Calculate total sparsity speculation with each bit-level sparsity with binary weight
        speculation = 0
        for i in range(self.xbit-1, -1, -1):
            speculation += sparsity_map[self.xbit-1-i] * (2 ** i)
        # Normalize estimation into (64, 3, 3) kernel size
        speculation = speculation * 64 * 3 * 3 / (kernel * kernel * channel)
        return speculation

    def _pac_calc(self, x_map, w_map):
        """
        Bit-wise PAC calculation for the given x_map and w_map.
        x_map: input activation bit map
        w_map: weight bit map
        return: multi-bit mac output activations by PAC
        """
        # Record input shape of x_map & w_map
        xbit, batch, x_channel, x_height, x_width = x_map.shape
        wbit, filter_num, w_channel, w_height, w_width = w_map.shape
        # Speculate output activation using bit sparsity if dynamic configuration is enabled
        if self.dynamic_config:
            # Speculate output activation using bit sparsity
            speculation = self._speculate_output(x_map, self.kernel_size_param, x_channel)
        # Initialize output tensors for exact and approximate computation
        digital = torch.zeros(batch, filter_num, device=self.device)
        pac = torch.zeros(batch, filter_num, device=self.device)
        # Extend dimension of x_map (input activation) and w_map (weight) with filter_num and batch respectively for parallel computation
        x_map = x_map.repeat(filter_num, 1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4, 5).to(torch.float32)
        w_map = w_map.repeat(batch, 1, 1, 1, 1, 1).permute(1, 0, 2, 3, 4, 5).to(torch.float32)
        if self.dynamic_config:
            # Compute method when threshold > th2
            if speculation > self.threshold[2]:
                # Digital computation
                for i in range(wbit-1, wbit-self.operand-1, -1):
                    for j in range(xbit-1, xbit-self.operand-1, -1):
                        partial_sum = torch.mul(w_map[wbit-1-i], x_map[xbit-1-j])
                        cycle_sum = torch.sum(partial_sum, (2, 3, 4))
                        # Weighted bit shift
                        digital += torch.mul(cycle_sum, 2.0 ** (i + j))
                # PAC computation
                for i in range(wbit-1, -1, -1):
                    for j in range(xbit-1, -1, -1):
                        if i > (wbit-self.operand-1) and j > (xbit-self.operand-1):
                            # Skip digital cycles
                            continue
                        # Compute weight and input activation bit sparsity
                        w_count = torch.sum(w_map[wbit-1-i], (2, 3, 4))
                        x_count = torch.sum(x_map[xbit-1-j], (2, 3, 4))
                        # Sparsity product
                        cycle_product = torch.mul(w_count, x_count)
                        # PAC MAC output
                        cycle_approx = torch.div(cycle_product, self.in_channels_param * self.kernel_size_param * self.kernel_size_param)
                        pac += torch.mul(cycle_approx, 2.0 ** (i + j))
                # Final multi-bit MAC output of combined digital and PAC computation
                output = digital + pac

            # Compute method when th1 < threshold <= th2
            elif speculation > self.threshold[1]:
                for i in range(wbit-1, wbit-self.operand-1, -1):
                    for j in range(xbit-1, xbit-self.operand-1, -1):
                        if (i + j) > 8:  # Skip w4x4
                            partial_sum = torch.mul(w_map[wbit-1-i], x_map[xbit-1-j])
                            cycle_sum = torch.sum(partial_sum, (2, 3, 4))
                            digital += torch.mul(cycle_sum, 2.0 ** (i + j))
                for i in range(wbit-1, -1, -1):
                    for j in range(xbit-1, -1, -1):
                        if i > (wbit-self.operand-1) and j > (xbit-self.operand-1):
                            if (i + j) > 8:  # Compute w4x4 with PAC
                                continue
                            else:
                                pass
                        w_count = torch.sum(w_map[wbit-1-i], (2, 3, 4))
                        x_count = torch.sum(x_map[xbit-1-j], (2, 3, 4))
                        cycle_product = torch.mul(w_count, x_count)
                        cycle_approx = torch.div(cycle_product, self.in_channels_param * self.kernel_size_param * self.kernel_size_param)
                        pac += torch.mul(cycle_approx, 2.0 ** (i + j))
                output = digital + pac

            # Compute method when th0 < threshold <= th1
            elif speculation > self.threshold[0]:
                for i in range(wbit-1, wbit-self.operand-1, -1):
                    for j in range(xbit-1, xbit-self.operand-1, -1):
                        if (i + j) > 9:  # Also skip w4x5, w5x4
                            partial_sum = torch.mul(w_map[wbit-1-i], x_map[xbit-1-j])
                            cycle_sum = torch.sum(partial_sum, (2, 3, 4))
                            digital += torch.mul(cycle_sum, 2.0 ** (i + j))
                for i in range(wbit-1, -1, -1):
                    for j in range(xbit-1, -1, -1):
                        if i > (wbit-self.operand-1) and j > (xbit-self.operand-1):
                            if (i + j) > 9:  # Also compute w4x5, w5x4 with PAC
                                continue
                            else:
                                pass
                        w_count = torch.sum(w_map[wbit-1-i], (2, 3, 4))
                        x_count = torch.sum(x_map[xbit-1-j], (2, 3, 4))
                        cycle_product = torch.mul(w_count, x_count)
                        cycle_approx = torch.div(cycle_product, self.in_channels_param * self.kernel_size_param * self.kernel_size_param)
                        pac += torch.mul(cycle_approx, 2.0 ** (i + j))
                output = digital + pac

            # Compute method when threshold <= th0
            elif speculation <= self.threshold[0]:
                for i in range(wbit-1, wbit-self.operand-1, -1):
                    for j in range(xbit-1, xbit-self.operand-1, -1):
                        if (i + j) > 10:  # Also skip w4x6, w5x5, w6x4
                            partial_sum = torch.mul(w_map[wbit-1-i], x_map[xbit-1-j])
                            cycle_sum = torch.sum(partial_sum, (2, 3, 4))
                            digital += torch.mul(cycle_sum, 2.0 ** (i + j))
                for i in range(wbit-1, -1, -1):
                    for j in range(xbit-1, -1, -1):
                        if i > (wbit-self.operand-1) and j > (xbit-self.operand-1):
                            if (i + j) > 10:  # Also compute w4x6, w5x5, w6x4 with PAC
                                continue
                            else:
                                pass
                        w_count = torch.sum(w_map[wbit-1-i], (2, 3, 4))
                        x_count = torch.sum(x_map[xbit-1-j], (2, 3, 4))
                        cycle_product = torch.mul(w_count, x_count)
                        cycle_approx = torch.div(cycle_product, self.in_channels_param * self.kernel_size_param * self.kernel_size_param)
                        pac += torch.mul(cycle_approx, 2.0 ** (i + j))
                output = digital + pac

        # Standard PACiM computation with dynamic configuration disabled
        else:
            # Digital computation
            for i in range(wbit-1, wbit-self.operand-1, -1):
                for j in range(xbit-1, xbit-self.operand-1, -1):
                    partial_sum = torch.mul(w_map[wbit-1-i], x_map[xbit-1-j])
                    cycle_sum = torch.sum(partial_sum, (2, 3, 4))
                    digital += torch.mul(cycle_sum, 2.0 ** (i + j))
            # PAC computation
            for i in range(wbit-1, -1, -1):
                for j in range(xbit-1, -1, -1):
                    if i > (wbit-self.operand-1) and j > (xbit-self.operand-1):
                        continue
                    w_count = torch.sum(w_map[wbit-1-i], (2, 3, 4))
                    x_count = torch.sum(x_map[xbit-1-j], (2, 3, 4))
                    cycle_product = torch.mul(w_count, x_count)
                    cycle_approx = torch.div(cycle_product, self.in_channels_param * self.kernel_size_param * self.kernel_size_param)
                    pac += torch.mul(cycle_approx, 2.0 ** (i + j))
            output = digital + pac

        return output

    def forward(self, input):
        """
        Forward call of PAConv2d with selective operation mode.
        input: input activation tensor
        return: output activation tensor
        """
        if self.mode == 'Train':    # Training mode with noise-aware training option
            w_quant = self._quantize_weight_train(self.weight)
            x_quant = self._quantize_feature_train(input)
            output = F.conv2d(x_quant,
                              w_quant,
                              self.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)
            # Add perturbation to output follows normal distribution with mean = 0 and standard deviation = trim_noise %
            perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.trim_noise * output
            output = output + perturb

        if self.mode == 'Inference':    # Inference mode that mimic UINT output with optional noise evaluation
            w_quant, w_scale, w_bias, w_sum = self._quantize_weight_infer(self.weight)
            x_quant, x_scale, x_bias, x_sum = self._quantize_feature_infer(input)[0:4]
            output = F.conv2d(x_quant,
                              w_quant,
                              bias=self.bias,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups)
            # Add perturbation to output follows normal distribution with mean = 0 and standard deviation = trim_noise %
            perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.trim_noise * output
            output = output + perturb
            # De-quantization to FP32
            batch, channel, height, width = output.shape
            w_sum = w_sum.repeat(batch, height, width, 1).permute(0, 3, 1, 2)
            x_sum = x_sum.repeat(channel, 1, 1, 1).permute(1, 0, 2, 3)
            output = output * (w_scale * x_scale) / ((2.0 ** self.wbit - 1.0) * (2.0 ** self.xbit - 1.0))
            output = output + x_bias * w_sum + w_bias * x_sum - w_bias * x_bias * self.in_channels_param * self.kernel_size_param * self.kernel_size_param
            # Compensation bias if needed
            if self.bias_param:
                x_height = output.shape[-2]
                x_width = output.shape[-1]
                bias_infer = self.bias.repeat(x_height, x_width, 1).permute(2, 0, 1)
                output += bias_infer

        if self.mode == 'Simulation':   # Bit-wise simulation using the framework
            weight, w_scale, w_bias, w_sum = self._quantize_weight_infer(self.weight)
            input, x_scale, x_bias, x_sum, x_zero = self._quantize_feature_infer(input)[0:5]
            # Record dimension of weight and input activation tensor
            num_w, channel_w, height_w, width_w = weight.shape
            batch_x, channel_x, height_x, width_x = input.shape
            # Manual padding zeros
            input = F.pad(input,
                          (self.padding_param, self.padding_param, self.padding_param, self.padding_param),
                          'constant',
                          x_zero)
            # Output activation tensor initialization
            height_o = math.floor((height_x - height_w + 2 * self.padding_param) / self.stride_param + 1)
            width_o = math.floor((width_x - width_w + 2 * self.padding_param) / self.stride_param + 1)
            output = torch.zeros(batch_x, num_w, height_o, width_o, device=self.device)
            # Decompose input bit map
            input = self._decompose_feature(input).to(dtype=torch.int8)
            # Decompose weight bit map
            weight_map = self._decompose_weight(weight).to(dtype=torch.int8)
            # Loop computation by striding
            for i in range(0, height_x - height_w + 2 * self.padding_param + 1, self.stride_param):
                for j in range(0, width_x - width_w + 2 * self.padding_param + 1, self.stride_param):
                    # Select computation field
                    field_x = input[:, :, :, i:i + height_w, j:j + width_w]
                    # Bit-wise computation on selected field
                    output[:, :, int(i / self.stride_param), int(j / self.stride_param)] = self._pac_calc(field_x, weight_map)
            # De-quantization to FP32
            w_sum = w_sum.repeat(batch_x, height_o, width_o, 1).permute(0, 3, 1, 2)
            x_sum = x_sum.repeat(num_w, 1, 1, 1).permute(1, 0, 2, 3)
            output = output * (w_scale * x_scale) / ((2.0 ** self.wbit - 1.0) * (2.0 ** self.xbit - 1.0))
            output = output + x_bias * w_sum + w_bias * x_sum - w_bias * x_bias * self.in_channels_param * self.kernel_size_param * self.kernel_size_param
            # Compensation bias if needed
            if self.bias_param:
                x_height = output.shape[-2]
                x_width = output.shape[-1]
                bias_sim = self.bias.repeat(x_height, x_width, 1).permute(2, 0, 1)
                output += bias_sim

        return output


class PALinear(nn.Linear):
    """
    Bit-wise simulation PALinear Module.
        in_features: input neurons
        out_features: output neurons
        bias: bias
        wbit: bit width of weight
        xbit: bit width of input activation
        mode: operation mode, e.g.: 'Train' or 'Inference' or 'Simulation'
        trim_noise: standard deviation applied for noise-aware training in 'Train' mode or roughly evaluate noise impact in 'Inference' mode
        device: 'cpu' or 'cuda' or 'mps' depend on your device
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 wbit=8,
                 xbit=8,
                 operand=4,
                 mode='Train',
                 trim_noise=0.0,
                 device='cpu'):

        assert mode in ['Train', 'Inference', 'Simulation'], "Invalid mode specified. Choose from 'Train', 'Inference', or 'Simulation'."

        super(PALinear, self).__init__(in_features,
                                       out_features,
                                       bias)
        self.reset_parameters()
        self.in_channel = in_features
        self.out_channel = out_features
        self.wbit = wbit
        self.xbit = xbit
        self.operand = operand
        self.bias_param = bias
        self.epsilon = 1e-7
        self.mode = mode
        self.trim_noise = trim_noise / 100
        self.device = device

    def _quantize_weight_train(self, input):
        """
        Quantize weight tensor in 'Train' mode.
        input: input weight tensor
        return: fake quantized weight tensor for training
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        w_bias = torch.min(input)
        input = torch.sub(input, w_bias)
        w_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / w_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.wbit - 1.0)) / (2.0 ** self.wbit - 1.0) # UINT Quantization
        return input * w_scale + w_bias

    def _quantize_feature_train(self, input):
        """
        Quantize input activation tensor in 'Train' mode.
        input: input activation tensor
        return: fake quantized input activation tensor for training
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_bias = torch.min(input)
        input = torch.sub(input, x_bias)
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0)) / (2.0 ** self.xbit - 1.0) # UINT Quantization
        return input * x_scale + x_bias

    def _quantize_weight_infer(self, input):
        """
        Quantize weight tensor in 'Inference' mode.
        input: input weight tensor
        return: quantized weight with UINT format for inference, scale & bias factor of weight, filter weight summation for dequantization
        """
        assert self.wbit > 1, "Bit width must be greater than 1."
        w_bias = torch.min(input)
        input = torch.sub(input, w_bias)
        w_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / w_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.wbit - 1.0)) # Quantization with UINT format
        # Calculate filter weight summation for dequantization
        fake_quant = torch.mul(input / (2.0 ** self.wbit - 1.0), w_scale)
        fake_quant = torch.add(fake_quant, w_bias)
        w_sum = torch.sum(fake_quant, dim=1)
        return input, w_scale, w_bias, w_sum

    def _quantize_feature_infer(self, input):
        """
        Quantize input activation tensor in 'Inference' mode.
        input: input activation tensor
        return: quantized input activation tensor with UINT format for inference, scale & bias factor of input activation, filter activation summation for dequantization
        """
        assert self.xbit > 1, "Bit width must be greater than 1."
        x_bias = torch.min(input)
        input = torch.sub(input, x_bias)
        x_scale = torch.max(input).detach() + self.epsilon
        input = Clamp.apply(input / x_scale, 0.0, 1.0) # range from 0~1
        input = Round.apply(input * (2.0 ** self.xbit - 1.0)) # Quantization with UINT format
        # Calculate filter activation summation for dequantization
        fake_quant = torch.mul(input / (2.0 ** self.xbit - 1.0), x_scale)
        fake_quant = torch.add(fake_quant, x_bias)
        x_sum = torch.sum(fake_quant, dim=1)
        return input, x_scale, x_bias, x_sum

    def _decompose_weight(self, input):
        """
        Decompose FP32 weight into UINT result tensor with '0' and '1'.
        input: input weight tensor in FP32
        return: decomposed UINT quantized tensor with '0' and '1' in shape (wbit, out_channel, in_channel)
        """
        # Create tensor to store decomposed UINT bit results
        w_map = torch.tensor([], device=self.device)
        # Loop to integrate remaining bits
        for i in range(self.wbit-1, -1, -1):
            w_map = torch.cat((w_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
            input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (wbit, out_channel, in_channel), e.g.: (8, 100, 4096)
        w_map = w_map.reshape(self.wbit, self.out_channel, self.in_channel)
        return w_map

    def _decompose_feature(self, input):
        """
        Decompose FP32 input activation into UINT result tensor with '0' and '1'.
        input: input activation tensor in FP32
        return: decomposed UINT quantized tensor with '0' and '1' in shape (xbit, batch, in_channel)
        """
        batch = input.shape[0]
        # Create tensor to store decomposed UINT bit results
        x_map = torch.tensor([], device=self.device)
        # Loop to integrate remaining bits
        for i in range(self.xbit-1, -1, -1):
            x_map = torch.cat((x_map, torch.div(input, 2.0 ** i, rounding_mode='floor')))
            input = torch.remainder(input, 2.0 ** i)
        # Reshape the tensor to (xbit, batch, in_channel), e.g.: (8, 4, 4096)
        x_map = x_map.reshape(self.xbit, batch, self.in_channel)
        return x_map

    def _pac_calc(self, x_map, w_map):
        """
        Bit-wise PAC calculation for the given x_map and w_map.
        x_map: input activation bit map
        w_map: weight bit map
        return: multi-bit mac output activations
        """
        # Record tensor shape parameters
        batch = x_map.shape[1]
        xbit = x_map.shape[0]
        wbit = w_map.shape[0]
        # Initialize tensors for digital and PAC computation
        digital = torch.zeros((batch, self.out_channel), device=self.device)
        pac = torch.zeros((batch, self.out_channel), device=self.device)
        # Extend dimension of x_map (input activation) and w_map (weight) with out_channel and batch respectively for parallel computation
        x_map = x_map.repeat(self.out_channel, 1, 1, 1).permute(1, 2, 3, 0).to(torch.float32)
        w_map = w_map.repeat(batch, 1, 1, 1).permute(1, 0, 3, 2).to(torch.float32)

        # Digital computation
        for i in range(wbit-1, wbit-self.operand-1, -1):
            for j in range(xbit-1, xbit-self.operand-1, -1):
                partial_sum = torch.mul(w_map[wbit-1-i], x_map[xbit-1-j])
                cycle_sum = torch.sum(partial_sum, dim=1)
                digital += torch.mul(cycle_sum, 2.0 ** (i + j))
        # PAC computation
        for i in range(wbit-1, -1, -1):
            for j in range(xbit-1, -1, -1):
                if i > (wbit-self.operand-1) and j > (xbit-self.operand-1):
                    continue
                w_count = torch.sum(w_map[wbit-1-i], dim=1)
                x_count = torch.sum(x_map[xbit-1-j], dim=1)
                cycle_product = torch.mul(w_count, x_count)
                cycle_approx = torch.div(cycle_product, self.in_channel)
                pac += torch.mul(cycle_approx, 2.0 ** (i + j))
        # Final multi-bit MAC output of combined digital and PAC computation
        output = digital + pac

        return output

    def forward(self, input):
        """
        Forward call of SimLinear with selective operation mode.
        input: input activation tensor
        return: output activation tensor
        """
        if self.mode == 'Train':    # Training mode with noise-aware training option
            w_quant = self._quantize_weight_train(self.weight)
            x_quant = self._quantize_feature_train(input)
            if self.bias_param: # Compensate bias if needed
                output = F.linear(x_quant, w_quant, self.bias)
            else:   # No bias linear layer
                output = F.linear(x_quant, w_quant)
            # Add perturbation to output follows normal distribution with mean = 0 and standard deviation = trim_noise %
            perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.trim_noise * output
            output = output + perturb

        if self.mode == 'Inference':    # Inference mode that mimic UINT output with optional noise evaluation
            w_quant, w_scale, w_bias, w_sum = self._quantize_weight_infer(self.weight)
            x_quant, x_scale, x_bias, x_sum = self._quantize_feature_infer(input)
            # Compute linear without bias
            output = F.linear(x_quant, w_quant)
            # Add perturbation to output follows normal distribution with mean = 0 and standard deviation = trim_noise %
            perturb = torch.tensor(0, device=self.device).repeat(*output.size()).float().normal_() * self.trim_noise * output
            output = output + perturb
            # De-quantization to FP32
            batch = output.shape[0]
            w_sum = w_sum.repeat(batch, 1).permute(0, 1)
            x_sum = x_sum.repeat(self.out_channel, 1).permute(1, 0)
            output = output * (w_scale * x_scale) / ((2.0 ** self.wbit - 1) * (2.0 ** self.xbit - 1))
            output = output + x_bias * w_sum + w_bias * x_sum - w_bias * x_bias * self.in_channel
            # Compensation bias if needed
            if self.bias_param:
                output += self.bias

        if self.mode == 'Simulation':   # Bit-wise simulation using the framework
            w_quant, w_scale, w_bias, w_sum = self._quantize_weight_infer(self.weight)
            x_quant, x_scale, x_bias, x_sum = self._quantize_feature_infer(input)
            # Decompose weight and input activation bit map
            w_map = self._decompose_weight(w_quant).to(dtype=torch.int8)
            x_map = self._decompose_feature(x_quant).to(dtype=torch.int8)
            # Compute output activations
            output = self._pac_calc(x_map, w_map)
            # De-quantization to FP32
            batch = output.shape[0]
            w_sum = w_sum.repeat(batch, 1).permute(0, 1)
            x_sum = x_sum.repeat(self.out_channel, 1).permute(1, 0)
            output = output * (w_scale * x_scale) / ((2.0 ** self.wbit - 1) * (2.0 ** self.xbit - 1))
            output = output + x_bias * w_sum + w_bias * x_sum - w_bias * x_bias * self.in_channel
            # Compensation bias if needed
            if self.bias_param:
                output += self.bias

        return output


if __name__ == '__main__':  # testbench

    setup_seed(6666)
    device = 'cpu'

    fake_conv_img = torch.randn((1, 64, 32, 32), device=device)
    fake_conv_weight = torch.randn((1, 64, 3, 3), device=device)
    fake_linear_feature = torch.randn(4, 4096)
    fake_linear_weight = torch.randn(100, 4096)
    fake_linear_bias = torch.randn(100)

    model_conv_train = PAConv2d(64,
                                1,
                                kernel_size=3,
                                stride=1,
                                padding=0,
                                bias=False,
                                wbit=8,
                                xbit=8,
                                operand=8,
                                dynamic_config=False,
                                threshold=(7000.0, 10000.0, 13000.0),
                                mode='Train',
                                trim_noise=0.0,
                                device='cpu')

    model_conv_infer = PAConv2d(64,
                                1,
                                kernel_size=3,
                                stride=1,
                                padding=0,
                                bias=False,
                                wbit=8,
                                xbit=8,
                                operand=8,
                                dynamic_config=False,
                                threshold=(7000.0, 10000.0, 13000.0),
                                mode='Inference',
                                trim_noise=0.0,
                                device='cpu')

    model_conv_sim = PAConv2d(64,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=0,
                              bias=False,
                              wbit=8,
                              xbit=8,
                              operand=8,
                              dynamic_config=False,
                              threshold=(7000.0, 10000.0, 13000.0),
                              mode='Simulation',
                              trim_noise=0.0,
                              device='cpu')

    model_conv_train._parameters['weight'] = fake_conv_weight
    model_conv_infer._parameters['weight'] = fake_conv_weight
    model_conv_sim._parameters['weight'] = fake_conv_weight

    output_conv_train = model_conv_train(fake_conv_img)
    output_conv_infer = model_conv_infer(fake_conv_img)
    output_conv_sim = model_conv_sim(fake_conv_img)

    train_to_infer_conv_error = output_conv_infer - output_conv_train
    train_to_infer_conv_error_perc = train_to_infer_conv_error / output_conv_sim
    infer_to_sim_conv_error = output_conv_sim - output_conv_infer
    infer_to_sim_conv_error_perc = infer_to_sim_conv_error / output_conv_infer

    print('Conv Layer: Train to Inference Error = {}, Sim to Inference Error = {}.'.format(train_to_infer_conv_error_perc, infer_to_sim_conv_error_perc))

    model_linear_train = PALinear(4096,
                                  100,
                                  bias=True,
                                  wbit=8,
                                  xbit=8,
                                  operand=8,
                                  mode='Train',
                                  trim_noise=0.0,
                                  device='cpu')

    model_linear_infer = PALinear(4096,
                                  100,
                                  bias=True,
                                  wbit=8,
                                  xbit=8,
                                  operand=8,
                                  mode='Inference',
                                  trim_noise=0.0,
                                  device='cpu')

    model_linear_sim = PALinear(4096,
                                100,
                                bias=True,
                                wbit=8,
                                xbit=8,
                                operand=8,
                                mode='Simulation',
                                trim_noise=0.0,
                                device='cpu')

    model_linear_train._parameters['weight'] = fake_linear_weight
    model_linear_infer._parameters['weight'] = fake_linear_weight
    model_linear_sim._parameters['weight'] = fake_linear_weight
    model_linear_train._parameters['bias'] = fake_linear_bias
    model_linear_infer._parameters['bias'] = fake_linear_bias
    model_linear_sim._parameters['bias'] = fake_linear_bias

    output_linear_train = model_linear_train(fake_linear_feature)
    output_linear_infer = model_linear_infer(fake_linear_feature)
    output_linear_sim = model_linear_sim(fake_linear_feature)

    train_to_infer_linear_error = output_linear_infer - output_linear_train
    train_to_infer_linear_error_perc = train_to_infer_linear_error / output_linear_sim
    infer_to_sim_linear_error = output_linear_sim - output_linear_infer
    infer_to_sim_linear_error_perc = infer_to_sim_linear_error / output_linear_infer

    print('Linear Layer: Train to Inference Error = {}, Sim to Inference Error = {}.'.format(train_to_infer_linear_error_perc, infer_to_sim_linear_error_perc))
