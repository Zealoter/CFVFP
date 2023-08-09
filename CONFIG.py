"""
# @Author: JuQi
# @Time  : 2023/3/10 19:33
# @E-mail: 18672750887@163.com
"""

"""
    'game'：The environment of the game
    'rm_mode'：Optional
        vanilla：Representing the original CFR，
        eta：Represents the weight when changing RM. If eta mode is selected, it is necessary to set rm_ Eta value
        br：best-response
    'is_rm_plus'：Whether the representative adopts CFR+
    'is_sampling_chance'：
        'all_sampling':Full sampling
        'no_sampling':No sampling
    'ave_mode' :
        'vanilla': Natural average
        'log': The logarithm of T
        'liner': Linearity of T
    'lr'：To be improved
    'log_mode':
        'exponential':
        'normal':
"""
ft_sampling_train_config = {
    'PCFR'                   : {
        'game'              : None,
        'rm_mode'           : 'br',  # CFR
        'rm_eta'            : 1,
        'is_rm_plus'        : False,
        'is_sampling_chance': 'all_sampling',
        'ave_mode'          : 'square',
        'log_mode'          : 'exponential',
        'op_env'            : 'PCFR'
    },
    # 'vanilla CFR': {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'no_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'log_mode'          : 'exponential'
    # },
    'CFR+'                   : {
        'game'              : None,
        'rm_mode'           : 'vanilla',  # CFR
        'rm_eta'            : 1,
        'is_rm_plus'        : True,
        'is_sampling_chance': 'no_sampling',
        'ave_mode'          : 'liner',
        'log_mode'          : 'exponential'
    },
    'External-Sampling-MCCFR': {
        'game'              : None,
        'rm_mode'           : 'vanilla',  # CFR
        'rm_eta'            : 1,
        'is_rm_plus'        : False,
        'is_sampling_chance': 'all_sampling',
        'ave_mode'          : 'square',
        'log_mode'          : 'exponential'
    },
    # 'External-Sampling-MCCFR-log'        : {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'log',
    #     'log_mode'          : 'exponential'
    # },
    # 'External-Sampling-MCCFR-liner': {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'liner',
    #     'log_mode'          : 'exponential'
    # },
    # 'br-MCCFR-liner'               : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'liner',
    #     'log_mode'          : 'exponential'
    # },
    # 'PCFR_log'                     : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'log',
    #     'log_mode'          : 'exponential',
    #     'op_env'            : 'PCFR'
    # },
    # 'PCFR'                     : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'log_mode'          : 'exponential',
    #     'op_env'            : 'PCFR'
    # },
    # 'External-Sampling-MCCFR-eta-fix': {
    #     'game'              : None,
    #     'rm_mode'           : 'eta_fix',  # CFR
    #     'rm_eta'            : 10,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'log_mode'          : 'exponential'
    # },
}

juqi_test_sampling_train_config = {
    'PMCCFR' : {
        'game'              : None,
        'rm_mode'           : 'br',
        'rm_eta'            : 1,
        'is_rm_plus'        : False,
        'is_sampling_chance': 'all_sampling',
        'ave_mode'          : 'log',
        'op_env'            : 'PCFR'
    },
    # 'PCFR'   : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'no_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'op_env'            : 'PCFR'
    # },
    # 'PCFR+'  : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : True,
    #     'is_sampling_chance': 'no_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'op_env'            : 'PCFR'
    # },
    # 'PMCCFR+': {
    #     'game'              : None,
    #     'rm_mode'           : 'br',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : True,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'op_env'            : 'PCFR'
    # },
    'WS': {
        'game'              : None,
        'rm_mode'           : 'br',
        'rm_eta'            : 1,
        'is_rm_plus'        : False,
        'is_sampling_chance': 'all_sampling',
        'ave_mode'          : 'log',
        'op_env'            : 'WS'
    },
    'CFR+': {
        'game'              : None,
        'rm_mode'           : 'vanilla',
        'rm_eta'            : 1,
        'is_rm_plus'        : True,
        'is_sampling_chance': 'no_sampling',
        'ave_mode'          : 'liner',
    },
    # 'CFR'                          : {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'no_sampling',
    #     'ave_mode'          : 'liner',
    # },
    # 'ES-MCCFR' : {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'square',
    # },

    # 'MIX-liner'                     : {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'square',
    #     'log_mode'          : 'exponential',
    #     'op_env'            : 'MIX'
    # },

    # 'External-Sampling-MCCFR-square': {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'square',
    #     'log_mode'          : 'exponential'
    # },

    # 'PCFR-liner'                    : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'liner',
    #     'log_mode'          : 'exponential',
    #     'op_env'            : 'PCFR'
    # },

}
