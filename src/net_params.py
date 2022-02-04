from collections import OrderedDict
from ConvRNN import CLSTM_cell, CGRU_cell
import numpy as np

dilation_scheme = [
        [1,1,1],
        [1,2,5],
        [1,2,3],
        ]

kernel_scheme = [
        [3,1,0], # kernel size, padding, conv type
        [3,1,1],
        [5,2,1],
        ]
# deconv
# out=(in-1)×stride-2p+k.


def get_filter(l):
    if l%2 == 0:
        return 4
    else:
        return 3

class NetParams():
    def __init__(self,config):
        self.filter_size = config.filter_size
        self.dilations = dilation_scheme[config.dilated]
        self.width_n_height = config.width_n_height
        
        self.layer_size = [self.width_n_height, int(np.ceil(self.width_n_height/2)), int(np.ceil(self.width_n_height/4))]
        self.layer_filter = [get_filter(self.layer_size[0]), get_filter(self.layer_size[1])]
        #self.kernel_size = config.kernel_size
        #self.image_width = int(np.ceil(pow(config.vec_len,0.5)))
        
        if config.method[0:3] == "org":
            org_type = int(config.method[4])
            s = kernel_scheme[org_type]
            self.encoder_params = [
                [
                    OrderedDict({'cross1_leaky_1': [1, 16, s[0], 1, s[1], s[2]]}), # in, out, kernel, stride, pad, kernel_type
                    OrderedDict({'cross1_leaky_1': [64, 64, s[0], 2, s[1], s[2]]}),
                    OrderedDict({'cross1_leaky_1': [96, 96, s[0], 2, s[1], s[2]]}),
                ],
            
                [
                    CLSTM_cell(shape=(32,32), input_channels=16, filter_size=self.filter_size, num_features=64,dilation=self.dilations[0]),
                    CLSTM_cell(shape=(16,16), input_channels=64, filter_size=self.filter_size, num_features=96,dilation=self.dilations[1]),
                    CLSTM_cell(shape=(8,8), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[2])
                ]
            ]
            
            self.decoder_params = [
                [
                    OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
                    OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
                    OrderedDict({
                        'conv3_leaky_1': [64, 16, 3, 1, 1],
                        'conv4_leaky_1': [16, 1, 1, 1, 0]
                    }),
                ],
            
                [
                    CLSTM_cell(shape=(8,8), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[2]),
                    CLSTM_cell(shape=(16,16), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[1]),
                    CLSTM_cell(shape=(32,32), input_channels=96, filter_size=self.filter_size, num_features=64,dilation=self.dilations[0]),
                ]
            ]
        elif config.method == "gru":
            self.encoder_params = [
                [
                    OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}), # in, out, kernal, stride, dilation
                    OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
                    OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
                ],
            
                [
                    CGRU_cell(shape=(self.layer_size[0],self.layer_size[0]), input_channels=16, filter_size=self.filter_size, num_features=64,dilation=self.dilations[0]),
                    CGRU_cell(shape=(self.layer_size[1],self.layer_size[1]), input_channels=64, filter_size=self.filter_size, num_features=96,dilation=self.dilations[1]),
                    CGRU_cell(shape=(self.layer_size[2],self.layer_size[2]), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[2])
                ]
            ]
            
            self.decoder_params = [
                [
                    OrderedDict({'deconv1_leaky_1': [96, 96, self.layer_filter[1], 2, 1]}),
                    OrderedDict({'deconv2_leaky_1': [96, 96, self.layer_filter[0], 2, 1]}),
                    OrderedDict({
                        'conv3_leaky_1': [64, 16, 3, 1, 1],
                        'conv4_leaky_1': [16, 1, 1, 1, 0]
                    }),
                ],
            
                [
                    CGRU_cell(shape=(self.layer_size[2],self.layer_size[2]), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[2]),
                    CGRU_cell(shape=(self.layer_size[1],self.layer_size[1]), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[1]),
                    CGRU_cell(shape=(self.layer_size[0],self.layer_size[0]), input_channels=96, filter_size=self.filter_size, num_features=64,dilation=self.dilations[0]),
                ]
            ]
        else:
            self.encoder_params = [
                [
                    OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}), # in, out, kernal, stride, dilation
                    OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
                    OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
                ],
            
                [
                    CLSTM_cell(shape=(self.layer_size[0],self.layer_size[0]), input_channels=16, filter_size=self.filter_size, num_features=64,dilation=self.dilations[0]),
                    CLSTM_cell(shape=(self.layer_size[1],self.layer_size[1]), input_channels=64, filter_size=self.filter_size, num_features=96,dilation=self.dilations[1]),
                    CLSTM_cell(shape=(self.layer_size[2],self.layer_size[2]), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[2])
                ]
            ]
            
            self.decoder_params = [
                [
                    OrderedDict({'deconv1_leaky_1': [96, 96, self.layer_filter[1], 2, 1]}),
                    OrderedDict({'deconv2_leaky_1': [96, 96, self.layer_filter[0], 2, 1]}),
                    OrderedDict({
                        'conv3_leaky_1': [64, 16, 3, 1, 1],
                        'conv4_leaky_1': [16, 1, 1, 1, 0]
                    }),
                ],
            
                [
                    CLSTM_cell(shape=(self.layer_size[2],self.layer_size[2]), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[2]),
                    CLSTM_cell(shape=(self.layer_size[1],self.layer_size[1]), input_channels=96, filter_size=self.filter_size, num_features=96,dilation=self.dilations[1]),
                    CLSTM_cell(shape=(self.layer_size[0],self.layer_size[0]), input_channels=96, filter_size=self.filter_size, num_features=64,dilation=self.dilations[0]),
                ]
            ]
        
            
#    convlstm_16_encoder_params = [
#            [
#                OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
#                OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
#                OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
#            ],
#        
#            [
#                CLSTM_cell(shape=(12,12), input_channels=16, filter_size=5, num_features=64),
#                CLSTM_cell(shape=(6,6), input_channels=64, filter_size=5, num_features=96),
#                CLSTM_cell(shape=(3,3), input_channels=96, filter_size=5, num_features=96)
#            ]
#        ]
#        
#        convlstm_16_decoder_params = [
#            [
#                OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
#                OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
#                OrderedDict({
#                    'conv3_leaky_1': [64, 16, 3, 1, 1],
#                    'conv4_leaky_1': [16, 1, 1, 1, 0]
#                }),
#            ],
#        
#            [
#                CLSTM_cell(shape=(3,3), input_channels=96, filter_size=5, num_features=96),
#                CLSTM_cell(shape=(6,6), input_channels=96, filter_size=5, num_features=96),
#                CLSTM_cell(shape=(12,12), input_channels=96, filter_size=5, num_features=64),
#            ]
#        ]
#    #46*46
#    convlstm_64_encoder_params = [
#        [
#            OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
#            OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
#            OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
#        ],
#    
#        [
#            CLSTM_cell(shape=(46,46), input_channels=16, filter_size=5, num_features=64),
#            CLSTM_cell(shape=(23,23), input_channels=64, filter_size=5, num_features=96),
#            CLSTM_cell(shape=(12,12), input_channels=96, filter_size=5, num_features=96)
#        ]
#    ]
#    
#    convlstm_64_decoder_params = [
#        [
#            OrderedDict({'deconv1_leaky_1': [96, 96, 3, 2, 1]}),
#            OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
#            OrderedDict({
#                'conv3_leaky_1': [64, 16, 3, 1, 1],
#                'conv4_leaky_1': [16, 1, 1, 1, 0]
#            }),
#        ],
#    
#        [
#            CLSTM_cell(shape=(12,12), input_channels=96, filter_size=5, num_features=96),
#            CLSTM_cell(shape=(23,23), input_channels=96, filter_size=5, num_features=96),
#            CLSTM_cell(shape=(46,46), input_channels=96, filter_size=5, num_features=64),
#        ]
#    ]
#    
#    convlstm_133_encoder_params = [
#        [
#            OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
#            OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
#            OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
#        ],
#    
#        [
#            CLSTM_cell(shape=(95,95), input_channels=16, filter_size=5, num_features=64),
#            CLSTM_cell(shape=(48,48), input_channels=64, filter_size=5, num_features=96),
#            CLSTM_cell(shape=(24,24), input_channels=96, filter_size=5, num_features=96)
#        ]
#    ]
#    
#    convlstm_133_decoder_params  = [
#        [
#            OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
#            OrderedDict({'deconv2_leaky_1': [96, 96, 3, 2, 1]}),
#            OrderedDict({
#                'conv3_leaky_1': [64, 16, 3, 1, 1],
#                'conv4_leaky_1': [16, 1, 1, 1, 0]
#            }),
#        ],
#    
#        [
#            CLSTM_cell(shape=(24,24), input_channels=96, filter_size=5, num_features=96),
#            CLSTM_cell(shape=(48,48), input_channels=96, filter_size=5, num_features=96),
#            CLSTM_cell(shape=(95,95), input_channels=96, filter_size=5, num_features=64),
#        ]
#    ]