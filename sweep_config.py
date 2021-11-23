import math
sweep_config = {
    'method': 'bayes'
    }
metric = {
    'name': 'test_accuracy',
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    }

sweep_config['parameters'] = parameters_dict
parameters_dict.update({
    'lr': {
        'distribution': 'uniform',
        'min': 0.0075,
        'max': 0.1
      },
          'batch_size': {
        'distribution': 'q_log_uniform',
        'q': 1,
        'min': math.log(64),
        'max': math.log(256),
      },
    'wd': {
        'distribution': 'uniform',
        'min': 5e-5,
        'max': 0.1,
      }
    })