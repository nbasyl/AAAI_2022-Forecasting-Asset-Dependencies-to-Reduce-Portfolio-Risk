# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from mlp import MLP
import numpy as np

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,device,input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.device = device
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts

        # transform_func_mlp 
  
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.normal.loc = self.normal.loc.cuda()
        self.normal.scale = self.normal.scale.cuda()

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch).cuda() * m + self.k
#        threshold_positions_if_in = threshold_positions_if_in.cuda()
        # print(top_values_flat.is_cuda)
        # print(threshold_positions_if_in.is_cuda)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        # print(clean_values.is_cuda)
        # print(threshold_if_in.is_cuda)
        # print(noise_stddev.is_cuda)

        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        # print(x)
        # print(x.shape)
        # print(True in torch.isnan(x))
        # print(self.w_gate)
        # print(True in torch.isnan(self.w_gate))
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # print(logits)
        # print(True in torch.isnan(logits))
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        # print(top_k_indices)
        # print(top_k_gates)
        # print(True in torch.isnan(top_k_gates))
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load



    def forward(self, x, train=True, loss_coef=1e-2, multiply_by_gates=True, visualization=False):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        # print(x.shape)
        # print(1 in torch.isnan(x))
        gates, load = self.noisy_top_k_gating(x, train)
        return_gates = gates
        # print("gates:", gates.shape)
        # print(gates)
        # calculate importance loss
        importance = gates.sum(0)
        # print(importance.shape)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # Initialize the dispatch
        # sort experts
        _gates_size = gates.size(0)
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, _expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        _batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        # print("gates:", gates[-1])
        # print((gates > 0).shape)
        _part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[_batch_index.flatten()]
        _nonzero_gates = torch.gather(gates_exp, 1, _expert_index)
        # END: Initialize the dispatch
        
        # dispatcher = self.dispatcher
        expert_inputs = None
        try:
            inp_exp = x[_batch_index].squeeze(1)
            # print("inp_exp: ", inp_exp.shape)
            # print("_part_sizes: ",_part_sizes)
            expert_inputs =  torch.split(inp_exp, _part_sizes, dim=0)
            gates = torch.split(_nonzero_gates, _part_sizes, dim=0)
        except Exception as e: 
            print(e)
#            print(gates.shape)
#            print(_part_sizes)
            exit(0)
            # inp_exp = x[_batch_index].squeeze(1)
            # if inp_exp.shape[0]%self.num_experts == 0:
            #   split_size = inp_exp.shape[0]//self.num_experts
            # else:
            #   split_size = inp_exp.shape[0]//self.num_experts + 1
            # expert_inputs =  torch.split(inp_exp, split_size, dim=0)
            # gates = torch.split(_nonzero_gates, split_size, dim=0)
        # END: dispatcher = self.dispatcher
    
        # gates = dispatcher.expert_to_gates()
        
        
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        
        # y = dispatcher.combine(expert_outputs)
        stitched = torch.cat(expert_outputs, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(_nonzero_gates)

        zeros = torch.zeros(_gates_size, expert_outputs[-1].size(1), requires_grad=True).cuda()
        # combine samples that have been processed by the same k experts
        y = zeros.index_add(0, _batch_index, stitched.float())
        # END: y = dispatcher.combine(expert_outputs)
        if visualization==True:
            return y, loss, return_gates
        else:
            return y, loss

