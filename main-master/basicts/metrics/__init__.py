from .corr import masked_corr
from .mae import masked_mae,masked_mae_bts
from .mape import masked_mape,masked_mape_bts
from .mse import masked_mse,masked_mse_bts
from .r_square import masked_r2
from .rmse import masked_rmse,masked_rmse_bts
from .smape import masked_smape
from .wape import masked_wape
from .mymetrick import masked_ae, masked_se, masked_ape

ALL_METRICS = {
            'MAEbts': masked_mae_bts,
            'MSEbts': masked_mse_bts,    
            'RMSEbts': masked_rmse_bts,
            'MAPEbts': masked_mape_bts,
            'WAPE': masked_wape,
            'MAEzh': masked_mae,
            'MSEzh': masked_mse,
            'RMSEzh': masked_rmse,
            'MAPEzh': masked_mape,
            'WAPEzh': masked_wape,
            'SMAPEzh': masked_smape,
            'R2zh': masked_r2,
            'CORRzh': masked_corr,
            'AE': masked_ae,
            'SE': masked_se,
            'APE': masked_ape
            }

__all__ = [
    'masked_mae_bts',
    'masked_mse_bts',
    'masked_rmse_bts',
    'masked_mape_bts',
    'masked_mae',
    'masked_mse',
    'masked_rmse',
    'masked_mape',
    'masked_wape',
    'masked_smape',
    'masked_r2',
    'masked_corr',
    'ALL_METRICS',
    'masked_ae',
    'masked_se',
    'masked_ape'
]