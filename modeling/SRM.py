import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvRNNCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvRNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)
        # self.conv1 = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)
        # self.conv2 = nn.Conv2d(in_channels=self.hidden_dim,
        #                       out_channels=self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)
        self.conv01 = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=self.input_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv02 = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=self.input_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.bn = nn.BatchNorm2d(self.hidden_dim, affine = True)
        self.relu = nn.ReLU()

        # print("input_dim",input_dim)
        # print("hidden_dim",hidden_dim)

    def forward(self, input_tensor, cur_state):
     

        # combined = torch.cat([input_tensor, cur_state], dim=1)  # concatenate along channel axis

        # x1 = self.conv1(combined)
        
        # # x1 = self.bn(x1)
        # # x1 = self.relu(x1)

        # x2 = self.conv2(x1)
        # # x2 = self.bn(x2)
        # x2 = self.relu(x2)

        # x1 = self.conv01(input_tensor)
        x1 = input_tensor
        x2 = self.conv02(cur_state)
        x = x1+x2
        # x = self.bn(x)
        x = self.relu(x) 


        # x1 = self.conv2(cur_state)
        # x2 = input_tensor + x1

        return x,x2

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv02.weight.device)

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, conv_direction='TD', seq_len = 20):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.conv_direction = conv_direction
        self.seq_len = seq_len
        # cell_list = []
       
        cur_input_dim = self.input_dim

        self.cell_list = ConvRNNCell(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dim[0],
                                        kernel_size=self.kernel_size[0],
                                        bias=self.bias)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, t, c, h, w = input_tensor.size()

        if self.conv_direction == 'DT':
            pass
    

        self.seq_len = t
        cur_state = self.cell_list.init_hidden(batch_size=b, image_size=(h, w))
        state_list = []
        for ii in range(self.seq_len):
            if self.conv_direction == 'DT':
                idx = self.seq_len - ii -1
            else:
                idx = ii
            cur_state, cur_output = self.cell_list(input_tensor=input_tensor[:, idx, :, :, :],  cur_state=cur_state)  
            state_list.append(cur_output)

        if self.conv_direction == 'DT':
            state_list.reverse()
        X = torch.stack(state_list, dim=1)
        return X

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class BSRM(nn.Module):
    def __init__(self, BatchNorm, input_dim, kernel_size, seq_len = 20):
        super(BSRM, self).__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
  
        self.RNN_TD = ConvLSTM(input_dim,input_dim,(1,kernel_size),1,
                                batch_first=True,
                                return_all_layers=True,
                                seq_len = seq_len,
                                conv_direction='TD')
       
        self.RNN_DT = ConvLSTM(input_dim,input_dim,(1,kernel_size),1,
                                batch_first=True,
                                return_all_layers=True,
                                seq_len = seq_len,
                                conv_direction='DT')
        
        self.last_conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=input_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              )


    def forward(self, x):

        b,c,h,w = x.size()
  
        x = torch.unsqueeze(x, dim=1) # B,1,C,H,W 
         
        x = x.permute(0, 3, 2, 1, 4)  # B,H,C,1,W
        x1 = self.RNN_TD(x)  
        x2 = self.RNN_DT(x)

        x = x1+x2
        x = x.permute(0, 3, 2, 1, 4)  # B,1,C,H,W
        x = torch.squeeze(x,dim=1)

        x = self.last_conv(x)

        # x = F.interpolate(x,size=[h,w], mode='bilinear', align_corners=True)

        return x

def build_BSRM(input_dim, kernel_size, BatchNorm):
    return BSRM(BatchNorm,input_dim=input_dim,kernel_size=kernel_size)


if __name__ == "__main__":
    import torch
    input = torch.rand(32, 128, 14, 14)    
    model = OrderConvTD(BatchNorm=nn.BatchNorm2d,input_dim=128,kernel_size=(1,5))
    output = model(input)
    
    print(output.shape)
