# Define the sweep behaviour
  
sweep_config = {
    'method': 'bayes', #bayes, grid, random
    'metric': {
      'name': 'uniform_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'batch_size': {
            'values': [512] #, 256, 384, 512, 640
        },
#        'dilated': {
#            'values': [0, 1, 2]
#        },
#        'filter_size': {
#            'values': [3, 5]
#        },
#        'lr': {
#            'min': 1e-5,
#            'max': 1e-3,
#        },
#        'lr_patience': {
#                'values':[4,7,10] 
#        },
#        'optimizer': {
#             "values"  : ['adam', 'sgd'] 
#        },
#        'input_gap':{
#            'values':[1,5,10,21]
#        },
        'num_experts':{
            'values': [8] #4, 6, 8, 10, 12
        },
        'top_k': {
            'values': [2] #2,3,4
        },
#        'method':{
#            'values': ["raw", "org_0", "org_1","org_2"]
#        },
#        'horizon': {
#            'values': [10, 21, 42]
#        },
    },
    "early_terminate":{
       "type": "hyperband",
       "s": 2,
       "eta": 3,
       "max_iter": 20,
   }
}
    
cluster_sweep_config = {
    'method': 'bayes', #bayes, grid, random
    'metric': {
      'name': 'uniform_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'batch_size': {
            'values': [256, 384, 512, 768]
        },
        'n_clusters': {
            'values': [4, 8, 12, 16]
        },
        'filter_size': {
            'values': [3, 5]
        },
    },
    "early_terminate":{
       "type": "hyperband",
       "s": 2,
       "eta": 3,
       "max_iter": 20,
   }
}
    
mlp_sweep_config = {
    'method': 'bayes', #bayes, grid, random
    'metric': {
      'name': 'uniform_loss',
      'goal': 'minimize'   
    },
    'parameters': {
#        'batch_size': {
#            'values': [16, 32, 64]
#        },
#        "top_k":{
#            'values': [4, 8]
#        },
#        'dilated': {
#            'values': [1, 2]
#        },
#        'filter_size': {
#            'values': [3, 5]
#        },
        'method':{
            'values': ["raw","mlp_0","mlp_1"]
        },
#        'lr': {
#            'min': 1e-5,
#            'max': 1e-3,
#        },
#        'lr_patience': {
#                'values':[4,7,10] 
#        },
#        'optimizer': {
#             "values"  : ['adam', 'sgd'] 
#        },
#        'input_gap':{
#            'values':[1,5,10,21]
#        },
#        'input_length':{
#            'values': [5, 8, 10]
#        },
#        'method':{
#            'values': ["mlp_0", "mlp_1", "moe_0","moe_1"]
#        },
#        'horizon': {
#            'values': [10, 21, 42]
#        },
    },
    "early_terminate":{
       "type": "hyperband",
       "s": 2,
       "eta": 3,
       "max_iter": 20,
   }
}
    
#org_sweep_config= {
#    'method': 'bayes', #bayes, grid, random
#    'metric': {
#      'name': 'val_loss',
#      'goal': 'minimize'   
#    },
#    'parameters': {
##        'batch_size': {
##            'values': [16, 32, 64]
##        },
#        'dilated': {
#            'values': [0, 1, 2]
#        },
#        'filter_size': {
#            'values': [3, 5]
#        },
#        'input_gap':{
#            'values':[1,5,10,21]
#        },
#        'input_length':{
#            'values': [5, 8, 10]
#        },
#    },
#    "early_terminate":{
#       "type": "hyperband",
#       "s": 2,
#       "eta": 3,
#       "max_iter": 20,
#   }
#}

moe_sweep_config = {
    'method': 'bayes', #bayes, grid, random
    'metric': {
      'name': 'uniform_loss',
      'goal': 'minimize'   
    },
    'parameters': {
#        'batch_size': {
#            'values': [16, 32, 64]
#        },
#        "top_k":{
#            'values': [4, 8]
#        },
#        'dilated': {
#            'values': [1, 2]
#        },
#        'filter_size': {
#            'values': [3, 5]
#        },
        'method':{
            'values': ["moe_0","moe_1"]
        },
#        'lr': {
#            'min': 1e-5,
#            'max': 1e-3,
#        },
#        'lr_patience': {
#                'values':[4,7,10] 
#        },
#        'optimizer': {
#             "values"  : ['adam', 'sgd'] 
#        },
#        'input_gap':{
#            'values':[1,5,10,21]
#        },
#        'input_length':{
#            'values': [5, 8, 10]
#        },
#        'method':{
#            'values': ["mlp_0", "mlp_1", "moe_0","moe_1"]
#        },
#        'horizon': {
#            'values': [10, 21, 42]
#        },
    },
    "early_terminate":{
       "type": "hyperband",
       "s": 2,
       "eta": 3,
       "max_iter": 20,
   }
}
    
#org_sweep_config= {
#    'method': 'bayes', #bayes, grid, random
#    'metric': {
#      'name': 'val_loss',
#      'goal': 'minimize'   
#    },
#    'parameters': {
##        'batch_size': {
##            'values': [16, 32, 64]
##        },
#        'dilated': {
#            'values': [0, 1, 2]
#        },
#        'filter_size': {
#            'values': [3, 5]
#        },
#        'input_gap':{
#            'values':[1,5,10,21]
#        },
#        'input_length':{
#            'values': [5, 8, 10]
#        },
#    },
#    "early_terminate":{
#       "type": "hyperband",
#       "s": 2,
#       "eta": 3,
#       "max_iter": 20,
#   }
#}

cov_sweep_config = {
    'method': 'bayes', #bayes, grid, random
    'metric': {
      'name': 'uniform_loss',
      'goal': 'minimize'   
    },
    'parameters': {
#        'batch_size': {
#            'values': [16, 32, 64]
#        },
        "num_experts": {
            'values': [4, 8]
        },
        "top_k":{
            'values': [2, 4, 8]
        },
#        'dilated': {
#            'values': [1, 2]
#        },
        'filter_size': {
            'values': [3, 5]
        },
#        'method':{
#            'values': ["moe_0","moe_1"]
#        },
#        'lr': {
#            'min': 1e-5,
#            'max': 1e-3,
#        },
#        'lr_patience': {
#                'values':[4,7,10] 
#        },
#        'optimizer': {
#             "values"  : ['adam', 'sgd'] 
#        },
        'input_gap':{
            'values':[1,5,10,21]
        },
        'input_length':{
            'values': [3, 6, 10, 12]
        },
#        'horizon': {
#            'values': [10, 21, 42]
#        },
    },
    "early_terminate":{
       "type": "hyperband",
       "s": 2,
       "eta": 3,
       "max_iter": 20,
   }
}
