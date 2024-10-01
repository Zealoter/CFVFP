test_sampling_train_config = {
    # 'CFVFP': {
    #     'game'         : None,
    #     'is_rm_plus'   : False,
    #     'sampling_mode': 'no_sampling',
    #     'ave_mode'     : 'vanilla',
    #     'op_env'       : 'CFVFP'
    # },
    'MCCFVFP'  : {
        'game'         : None,
        'rm_mode'      : 'br',
        'rm_eta'       : 1,
        'is_rm_plus'   : False,
        'sampling_mode': 'sampling',
        'ave_mode'     : 'vanilla',
        'op_env'       : 'CFVFP'
    },
    # 'CFR'    : {
    #     'game'         : None,
    #     'is_rm_plus'   : False,
    #     'sampling_mode': 'no_sampling',
    #     'ave_mode'     : 'vanilla',
    # },
    # 'CFR+'    : {
    #     'game'         : None,
    #     'is_rm_plus'   : True,
    #     'sampling_mode': 'no_sampling',
    #     'ave_mode'     : 'square',
    # },
    'ES-MCCFR'    : {
        'game'         : None,
        'is_rm_plus'   : False,
        'sampling_mode': 'sampling',
        'ave_mode'     : 'vanilla',
    },

}
