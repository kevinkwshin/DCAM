from data import *
from main import *

path = '../EKG_PVC/20230203_MHATEST/featureLength1280_lossFnbceloss_modelNameefficientnet-b0_normlayer_se_moduleacm_segheadModuleSE_skipModuleACM4_0_BOTTOM5_supervisionTYPE1_trainaugNEUROKIT2_upsamplepixelshuffle_dataSeed1_srTarget250_samplerTrue_inChannels1_outChannels2_dataNormzscore/weight/best_val_loss.ckpt'
test(path, True)