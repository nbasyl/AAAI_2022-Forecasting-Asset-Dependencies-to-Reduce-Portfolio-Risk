from moe import MoE
from mlp import MLP
from torch import nn
import torch.nn.functional as F
import torch
from utils import make_layers
from modules_3d import *

moe = ((1,10,[3,3,3],1,[1,1,1]) , (10,1,[3,3,3],1,[0,1,1]))
d_T = ((1,10,[3,3,3],1,[1,1,1]) , (10,1,[3,3,3],1,[1,1,1]))

class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError

class ED_RAW(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()        
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output
    
class ED_Tran(nn.Module):
    def __init__(self, encoder, decoder, input_length, image_width):
        super().__init__()        
        self.encoder = encoder
        self.decoder = decoder
        self.relu =  nn.ReLU()
        
#        size = input_length * image_width * image_width
        size = image_width ** 2
        self.T_net = MLP(size, size, 2*size)
    
    def forward(self, input):
        shape = input.shape
        x = input.view(shape[0]*shape[1], -1)
        x = self.T_net(x)
        x = x.view(shape)

        state = self.encoder(x)
        output = self.decoder(state)
        return output

class ED_simple_MOE(nn.Module):

    def __init__(self, encoder, decoder, input_length, image_width, device, num_experts = 4, top_k = 1, noisy_gating = True):
        super().__init__()
        # self.batch_size = batch_size
        # self.image_width = image_width 

        self.device = device

        kernel_2 = moe[1][2]
        kernel_2[0] = input_length
        self.conv3d_1 = nn.Conv3d(moe[0][0],moe[0][1],moe[0][2],stride = moe[0][3],padding=moe[0][4])
        self.conv3d_2 = nn.Conv3d(moe[1][0],moe[1][1],kernel_2,stride = moe[1][3],padding=moe[1][4])

        self.relu =  nn.ReLU()
        self.moe_transform = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_add = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, train=True):

        # conv3d to condense input from (batchsize, Seq, C, W, H) to the shape of (batchsize,W,H)
        # input shape(B,S,C,H,W) but conv3d accept (B,C,S,H,W)
        x = torch.transpose(input, 1, 2)

        x = self.conv3d_1(x)
        x = self.relu(x)
        x = self.conv3d_2(x)
        x = self.relu(x)

        # Reshape (batchsize, W, H) to (batchsize, W*H*C) as MoE only accept input shape (Batchsize, inputsize)
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        
        # Feed to MoE
        transform_func, aux_loss = self.moe_transform(x, train=train)
        add_func, add_aux_loss = self.moe_add(x, train=train)
        

        shape = input.shape
        # MoE will have output shape (Batchsize, outputsize) then reshape it to (Batchsize, W,H) in order to do brodcasting have to reshape (B,1,1,W,H)
        transform_func = transform_func.view(shape[0],1,shape[2],shape[3],shape[4])
        #print(transform_func.shape)
        add_func = add_func.view(shape[0],1,shape[2],shape[3],shape[4])
        # new_input = input * MoE_output
        identity = input
        new_input = (input * transform_func + add_func)

        state = self.encoder(new_input)
        output = self.decoder(state)
        return output, (aux_loss+add_aux_loss)
    
class ED_simple_MLP(nn.Module):

    def __init__(self, encoder, decoder, input_length, image_width):
        super().__init__()
        # self.batch_size = batch_size
        # self.image_width = image_width 
        # self.device = device

        kernel_2 = moe[1][2]
        kernel_2[0] = input_length
        self.conv3d_1 = nn.Conv3d(moe[0][0],moe[0][1],moe[0][2],stride = moe[0][3],padding=moe[0][4])
        self.conv3d_2 = nn.Conv3d(moe[1][0],moe[1][1],kernel_2,stride = moe[1][3],padding=moe[1][4])
        self.relu = nn.ReLU()
        self.mlp1 = MLP(image_width**2,image_width**2,2*(image_width**2))
        self.mlp2 = MLP(image_width**2,image_width**2,2*(image_width**2))

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):

        # conv3d to condense input from (batchsize, Seq, C, W, H) to the shape of (batchsize,W,H)
        # input shape(B,S,C,H,W) but conv3d accept (B,C,S,H,W)
        x = torch.transpose(input, 1, 2)
        x = self.conv3d_1(x)
        x = self.relu(x)
        x = self.conv3d_2(x)
        x = self.relu(x)
        # Reshape (batchsize, W, H) to (batchsize, W*H*C) as MoE only accept input shape (Batchsize, inputsize)
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        
        mul_func = self.mlp1(x)
        add_func = self.mlp2(x)
        shape = input.shape
        mul_func = mul_func.view(shape[0],1,shape[2],shape[3],shape[4])
        add_func = add_func.view(shape[0],1,shape[2],shape[3],shape[4])

        # new_input = input * MoE_output
        new_input = (input * mul_func + add_func)

        state = self.encoder(new_input)
        output = self.decoder(state)
        return output


class ED_QUAD_MOE(nn.Module):

    def __init__(self, encoder, decoder, input_length, image_width, device, num_experts = 4, top_k = 1, noisy_gating = True):
        super().__init__()
        # self.batch_size = batch_size
        # self.image_width = image_width 
        self.device = device
        
        kernel_2 = moe[1][2]
        kernel_2[0] = input_length
        self.conv3d_1 = nn.Conv3d(moe[0][0],moe[0][1],moe[0][2],stride = moe[0][3],padding=moe[0][4])
        self.conv3d_2 = nn.Conv3d(moe[1][0],moe[1][1],kernel_2,stride = moe[1][3],padding=moe[1][4])

        self.relu =  nn.ReLU()
        self.moe_transform = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_add = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_quad = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.encoder = encoder
        self.decoder = decoder
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, train=True, visualization=False):

        # conv3d to condense input from (batchsize, Seq, C, W, H) to the shape of (batchsize,W,H)
        # input shape(B,S,C,H,W) but conv3d accept (B,C,S,H,W)
        x = torch.transpose(input, 1, 2)
        # print(True in torch.isnan(x))
#        print(x.shape)
        
        x = self.conv3d_1(x)
#        print(x.shape)
        x = self.relu(x)
        x = self.conv3d_2(x)
#        print(x.shape)
        x = self.relu(x)
        # Reshape (batchsize, W, H) to (batchsize, W*H*C) as MoE only accept input shape (Batchsize, inputsize)
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        
#        print(x.shape)
        # Feed to MoE
        if visualization == True:
            transform_func, aux_loss, moe_transform_gates = self.moe_transform(x, train=train, visualization=True)
            add_func, add_aux_loss, moe_add_gates = self.moe_add(x, train=train, visualization=True)
            quad_func, quad_aux_loss, moe_quad_gates = self.moe_quad(x, train = train, visualization=True)
        
        else:
            transform_func, aux_loss = self.moe_transform(x, train=train)
            add_func, add_aux_loss = self.moe_add(x, train=train)
            quad_func, quad_aux_loss = self.moe_quad(x, train = train)
        

        shape = input.shape
        # MoE will have output shape (Batchsize, outputsize) then reshape it to (Batchsize, W,H) in order to do brodcasting have to reshape (B,1,1,W,H)
        transform_func = transform_func.view(shape[0],1,shape[2],shape[3],shape[4])
        add_func = add_func.view(shape[0],1,shape[2],shape[3],shape[4])
        quad_func = quad_func.view(shape[0],1,shape[2],shape[3],shape[4])
        # new_input = input * MoE_output
#        new_input = self.sigmoid(input*input*quad_func + input * transform_func + add_func)
        output = (input*input*quad_func + input * transform_func + add_func)

        output = self.encoder(output)
        output = self.decoder(output)
        output = self.sigmoid(output)
        if visualization == True:
            return output, torch.sum(aux_loss+add_aux_loss+quad_aux_loss), moe_transform_gates, moe_add_gates, moe_quad_gates
        else:
            return output, torch.sum(aux_loss+add_aux_loss+quad_aux_loss)
    
    
class ED_CUBIC_MOE(nn.Module):

    def __init__(self, encoder, decoder, input_length, image_width, device, num_experts = 4, top_k = 1, noisy_gating = True):
        super().__init__()
        # self.batch_size = batch_size
        # self.image_width = image_width 
        self.device = device
        
        kernel_2 = moe[1][2]
        kernel_2[0] = input_length
        self.conv3d_1 = nn.Conv3d(moe[0][0],moe[0][1],moe[0][2],stride = moe[0][3],padding=moe[0][4])
        self.conv3d_2 = nn.Conv3d(moe[1][0],moe[1][1],kernel_2,stride = moe[1][3],padding=moe[1][4])

        self.relu =  nn.ReLU()
        
        self.moe_transform = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_add = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_quad = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_cubic = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.encoder = encoder
        self.decoder = decoder
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, train=True):

        # conv3d to condense input from (batchsize, Seq, C, W, H) to the shape of (batchsize,W,H)
        # input shape(B,S,C,H,W) but conv3d accept (B,C,S,H,W)
        x = torch.transpose(input, 1, 2)

        x = self.conv3d_1(x)
        x = self.relu(x)
        x = self.conv3d_2(x)
        x = self.relu(x)
        # Reshape (batchsize, W, H) to (batchsize, W*H*C) as MoE only accept input shape (Batchsize, inputsize)
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        
        # Feed to MoE
        transform_func, aux_loss = self.moe_transform(x, train=train)
        add_func, add_aux_loss = self.moe_add(x, train=train)
        quad_func, quad_aux_loss = self.moe_quad(x, train = train)
        cubic_func, cubic_aux_loss = self.moe_cubic(x, train = train)
        
        shape = input.shape
        # MoE will have output shape (Batchsize, outputsize) then reshape it to (Batchsize, W,H) in order to do brodcasting have to reshape (B,1,1,W,H)
        transform_func = transform_func.view(shape[0],1,shape[2],shape[3],shape[4])
        add_func = add_func.view(shape[0],1,shape[2],shape[3],shape[4])
        quad_func = quad_func.view(shape[0],1,shape[2],shape[3],shape[4])
        cubic_func = cubic_func.view(shape[0],1,shape[2],shape[3],shape[4])
        # new_input = input * MoE_output
#        new_input = self.sigmoid(input*input*quad_func + input * transform_func + add_func)
        output = (input*input*input*cubic_func+input*input*quad_func + input * transform_func + add_func)

        output = self.encoder(output)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, torch.sum(aux_loss+add_aux_loss+quad_aux_loss+cubic_aux_loss)
    
class ED_Fourth_MOE(nn.Module):

    def __init__(self, encoder, decoder, input_length, image_width, device, num_experts = 4, top_k = 1, noisy_gating = True):
        super().__init__()
        # self.batch_size = batch_size
        # self.image_width = image_width 
        self.device = device
        
        kernel_2 = moe[1][2]
        kernel_2[0] = input_length
        self.conv3d_1 = nn.Conv3d(moe[0][0],moe[0][1],moe[0][2],stride = moe[0][3],padding=moe[0][4])
        self.conv3d_2 = nn.Conv3d(moe[1][0],moe[1][1],kernel_2,stride = moe[1][3],padding=moe[1][4])

        self.relu =  nn.ReLU()
        
        self.moe_transform = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_add = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_quad = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_cubic = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.moe_fourth = MoE(self.device,image_width**2,image_width**2,num_experts=num_experts,\
                                 hidden_size = 2*(image_width**2),k=top_k,noisy_gating = noisy_gating)
        self.encoder = encoder
        self.decoder = decoder
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, train=True):

        # conv3d to condense input from (batchsize, Seq, C, W, H) to the shape of (batchsize,W,H)
        # input shape(B,S,C,H,W) but conv3d accept (B,C,S,H,W)
        x = torch.transpose(input, 1, 2)

        x = self.conv3d_1(x)
        x = self.relu(x)
        x = self.conv3d_2(x)
        x = self.relu(x)
        # Reshape (batchsize, W, H) to (batchsize, W*H*C) as MoE only accept input shape (Batchsize, inputsize)
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        
        # Feed to MoE
        transform_func, aux_loss = self.moe_transform(x, train=train)
        add_func, add_aux_loss = self.moe_add(x, train=train)
        quad_func, quad_aux_loss = self.moe_quad(x, train = train)
        cubic_func, cubic_aux_loss = self.moe_cubic(x, train = train)
        fourth_func, fourth_aux_loss = self.moe_fourth(x, train = train)
        
        shape = input.shape
        # MoE will have output shape (Batchsize, outputsize) then reshape it to (Batchsize, W,H) in order to do brodcasting have to reshape (B,1,1,W,H)
        transform_func = transform_func.view(shape[0],1,shape[2],shape[3],shape[4])
        add_func = add_func.view(shape[0],1,shape[2],shape[3],shape[4])
        quad_func = quad_func.view(shape[0],1,shape[2],shape[3],shape[4])
        cubic_func = cubic_func.view(shape[0],1,shape[2],shape[3],shape[4])
        fourth_func = fourth_func.view(shape[0],1,shape[2],shape[3],shape[4])
        # new_input = input * MoE_output
#        new_input = self.sigmoid(input*input*quad_func + input * transform_func + add_func)
        output = (input*input*input*input*fourth_func+input*input*input*cubic_func+input*input*quad_func + input * transform_func + add_func)

        output = self.encoder(output)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, torch.sum(aux_loss+add_aux_loss+quad_aux_loss+cubic_aux_loss+fourth_aux_loss)
    
class ED_QUAD_MLP(nn.Module):

    def __init__(self, encoder, decoder, input_length, image_width):
        super().__init__()
        # self.batch_size = batch_size
        # self.image_width = image_width 
        # self.device = device

        kernel_2 = moe[1][2]
        kernel_2[0] = input_length
        self.conv3d_1 = nn.Conv3d(moe[0][0],moe[0][1],moe[0][2],stride = moe[0][3],padding=moe[0][4])
        self.conv3d_2 = nn.Conv3d(moe[1][0],moe[1][1],kernel_2,stride = moe[1][3],padding=moe[1][4])
        self.relu = nn.ReLU()
        self.mlp1 = MLP(image_width**2,image_width**2,2*(image_width**2))
        self.mlp2 = MLP(image_width**2,image_width**2,2*(image_width**2))
        self.mlp3 = MLP(image_width**2,image_width**2,2*(image_width**2))

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):

        # conv3d to condense input from (batchsize, Seq, C, W, H) to the shape of (batchsize,W,H)
        # input shape(B,S,C,H,W) but conv3d accept (B,C,S,H,W)
        x = torch.transpose(input, 1, 2)
        x = self.conv3d_1(x)
        x = self.relu(x)
        x = self.conv3d_2(x)
        x = self.relu(x)
        # Reshape (batchsize, W, H) to (batchsize, W*H*C) as MoE only accept input shape (Batchsize, inputsize)
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        
        
        
        mul_func = self.mlp1(x)
        add_func = self.mlp2(x)
        quad_func = self.mlp3(x)
        shape = input.shape
        mul_func = mul_func.view(shape[0],1,shape[2],shape[3],shape[4])
        add_func = add_func.view(shape[0],1,shape[2],shape[3],shape[4])
        quad_func = quad_func.view(shape[0],1,shape[2],shape[3],shape[4])

        # new_input = input * MoE_output
        identity = input
        new_input = (input*input*quad_func + input * mul_func + add_func)

        state = self.encoder(new_input)
        output = self.decoder(state)
        return output
    
#class POS_MOE(nn.Module):
#
#    def __init__(self, encoder, decoder, input_length, image_width, device, num_experts = 4, top_k = 1, noisy_gating = True):
#        super().__init__()
#        # self.batch_size = batch_size
#        # self.image_width = image_width 
#        self.device = device
#        
#        kernel_2 = moe[1][2]
#        kernel_2[0] = input_length
#        self.conv3d_1 = nn.Conv3d(moe[0][0],moe[0][1],moe[0][2],stride = moe[0][3],padding=moe[0][4])
#        self.conv3d_2 = nn.Conv3d(moe[1][0],moe[1][1],kernel_2,stride = moe[1][3],padding=moe[1][4])
#
#        self.relu =  nn.ReLU()
#        
#        self.moe_transform = MoE(self.device,image_width**2,image_width**4,num_experts=num_experts,\
#                                 hidden_size = image_width**4,k=top_k,noisy_gating = noisy_gating)
#        
#        self.encoder = encoder
#        self.decoder = decoder
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, input, train=True):
#
#        # conv3d to condense input from (batchsize, Seq, C, W, H) to the shape of (batchsize,W,H)
#        # input shape(B,S,C,H,W) but conv3d accept (B,C,S,H,W)
#        x = torch.transpose(input, 1, 2)
#
#        x = self.conv3d_1(x)
#        x = self.relu(x)
#        x = self.conv3d_2(x)
#        x = self.relu(x)
#        # Reshape (batchsize, W, H) to (batchsize, W*H*C) as MoE only accept input shape (Batchsize, inputsize)
#        x_shape = x.shape
#        x = x.reshape(x_shape[0], -1)
#        
#        # Feed to MoE
#        transform_func, aux_loss = self.moe_transform(x, train=train)
#        shape = input.shape
#        # MoE will have output shape (Batchsize, outputsize) then reshape it to (Batchsize, W,H) in order to do brodcasting have to reshape (B,1,1,W,H)
#        transform_func = transform_func.view(shape[0],1,shape[2],shape[3]**2,shape[4]**2)
#        input = input.view(shape[0],shape[1],shape[2],shape[3]**2,1)
#        # new_input = input * MoE_output
##        new_input = self.sigmoid(input*input*quad_func + input * transform_func + add_func)
#        output = torch.matmul(transform_func, input)
#        output = output.view(shape[0],shape[1],shape[2],shape[3],shape[4])
#
#        output = self.encoder(output)
#        output = self.decoder(output)
#        output = self.sigmoid(output)
#        return output,aux_loss
    
    
    
class Conv3d(nn.Module):
      def __init__(self,input_length):
          super().__init__()
          self.mod = nn.Sequential(
            	nn.Conv3d(1, 8, kernel_size = (3,3,3), padding = (1,1,1)),
            	nn.BatchNorm3d(8),
            	nn.ReLU(),
            
            	Conv3dBlock(8), Conv3dBlock(8), 
                    Conv3dDownsample(8),
                    Conv3dBlock(16), Conv3dBlock(16), 
            	Conv3dDownsample(16),
                    Conv3dBlock(32), Conv3dBlock(32), 
                    ConvTranspose3dBlock(32), ConvTranspose3dBlock(32), 
                    ConvTranspose3dUpsample(32),
                    ConvTranspose3dBlock(16), ConvTranspose3dBlock(16), 
                    ConvTranspose3dUpsample(16,2),
            	ConvTranspose3dBlock(8), ConvTranspose3dBlock(8), 
            
                    nn.Conv3d(8, 1, kernel_size = (1,1,1)),
                    nn.Sigmoid(),
        )
    
      def forward(self, frames_input):
          frames_input = torch.transpose(frames_input, 1, 2)
          frames_output = self.mod(frames_input)
#          print(frames_output.shape)
          frames_output = frames_output[:,:,0:1,...]
#          print(frames_output.shape)
          return frames_output
      
class Conv3d_1(nn.Module):
      def __init__(self,input_length):
          super().__init__()
          self.mod = nn.Sequential(
            	nn.Conv3d(1, 8, kernel_size = (3,3,3), padding = (1,1,1)),
            	nn.BatchNorm3d(8),
            	nn.ReLU(),
            
            	Conv3dBlock(8), Conv3dBlock(8), 
                    Conv3dDownsample(8),
                    Conv3dBlock(16), Conv3dBlock(16), 
            	Conv3dDownsample(16),
                    Conv3dBlock(32), Conv3dBlock(32), 
                    ConvTranspose3dBlock(32), ConvTranspose3dBlock(32), 
                    ConvTranspose3dUpsample(32),
                    ConvTranspose3dBlock(16), ConvTranspose3dBlock(16), 
                    ConvTranspose3dUpsample(16,2),
            	ConvTranspose3dBlock(8), ConvTranspose3dBlock(8), 
            
                    nn.Conv3d(8, 1, kernel_size = (1,1,1)),
            nn.Conv3d(1, 1, kernel_size = (input_length,3,3), padding = (0,1,1)),
            nn.Sigmoid(),
        )
    
      def forward(self, frames_input):
          frames_input = torch.transpose(frames_input, 1, 2)
          frames_output = self.mod(frames_input)
#          print(frames_output.shape)
          return frames_output
      
class E3D(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)
#        print(state.shape)
        output = self.decoder(state)
        return output


