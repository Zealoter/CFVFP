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
    'log_mode':
        'exponential':
        'normal':
"""
test_sampling_train_config = {
    'PMCCFR' : {
        'game'              : None,
        'rm_mode'           : 'br',
        'rm_eta'            : 1,
        'is_rm_plus'        : False,
        'is_sampling_chance': 'all_sampling',
        'ave_mode'          : 'log',
        'op_env'            : 'PCFR'
    },
    'CFR+': {
        'game'              : None,
        'rm_mode'           : 'vanilla',
        'rm_eta'            : 1,
        'is_rm_plus'        : True,
        'is_sampling_chance': 'no_sampling',
        'ave_mode'          : 'liner',
    },
    'CFR'                          : {
        'game'              : None,
        'rm_mode'           : 'vanilla',
        'rm_eta'            : 1,
        'is_rm_plus'        : False,
        'is_sampling_chance': 'no_sampling',
        'ave_mode'          : 'liner',
    },
    'ES-MCCFR' : {
        'game'              : None,
        'rm_mode'           : 'vanilla',
        'rm_eta'            : 1,
        'is_rm_plus'        : False,
        'is_sampling_chance': 'all_sampling',
        'ave_mode'          : 'square',
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
    # 'WS': {
    #     'game'              : None,
    #     'rm_mode'           : 'br',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'is_sampling_chance': 'all_sampling',
    #     'ave_mode'          : 'log',
    #     'op_env'            : 'WS'
    # },
}
