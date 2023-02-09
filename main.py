import os, sys, shutil
os.environ["HTTP_PROXY"] = "http://192.168.45.100:3128"
os.environ["HTTPS_PROXY"] = "http://192.168.45.100:3128"
import bagua_core; bagua_core.install_deps()

# export http_proxy=http://192.168.45.100:3128
# export https_proxy=https://192.168.45.100:3128

# !pip install monai neurokit2 wfdb monai pytorch_lightning==1.7.7 wandb libauc==1.2.0 --upgrade --quiet
# os.system('pip install monai neurokit2 wfdb monai pytorch_lightning==1.7.7 wandb libauc==1.2.0 --upgrade --quiet')

gpus= "0,1,2,3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["WANDB_API_KEY"] = '6cd6a2f58c8f4625faaea5c73fe110edab2be208'

from utils import *
from models import *
from data import *
from losses import *
import models.nets as nets

run_command('nvcc -V')
run_command('nvidia-smi')
# run_command('pip install monai neurokit2 wfdb monai pytorch_lightning==1.7.7 wandb libauc==1.2.0 --upgrade --quiet')
device = get_device()

NUM_WORKERS = os.cpu_count()
print("Number of workers:", NUM_WORKERS)
# print('multiprocessing.cpu_count()', multiprocessing.cpu_count())
print('cuda.is_available', torch.cuda.is_available())
# print(device)
# print_config()
        
config_defaults = dict(
    dataSeed = 1,
    srTarget = 250,
    featureLength = 1280,
    sampler = True, # True, False
    inChannels = 1,
    outChannels = 2,
    dataNorm ='zscoreO', # zscoreI, zscoreO, minmaxI
    
    project = 'PVC_NET',  # this is cutoff line of path_logRoot
    
    modelName='efficientnet-b2', # 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
    norm = 'instance', # 'instance', 'batch', 'group', 'layer'
    upsample = 'pixelshuffle', #'pixelshuffle', # 'nontrainable'
    supervision = "TYPE1", #'NONE', 'TYPE1', 'TYPE2'
    skipModule = "SE_BOTTOM5", 
    segheadModule = "SE",
    trainaug = 'NEUROKIT2',

    path_logRoot = '20230207_BASE',
    spatial_dims = 1,
    learning_rate = 4e-3,
    batch_size = 512, # 256
    dropout = 0.01,
    thresholdRPeak = 0.7,
    skipASPP = "NONE",
    lossFn = 'BCE',
    se = 'se',
    mtl = 'NONE'
)

def train():
    wandb.init(config=config_defaults)
    hyperparameters = dict(wandb.config)
    
    set_seed()
    model = PVC_NET(hyperparameters)
    
    classes = model.hyperparameters['outChannels']
    srTarget = model.hyperparameters['srTarget']
    featureLength = model.hyperparameters['featureLength']   
    dataNorm = model.hyperparameters['dataNorm']

    train_files = glob('dataset/MIT-BIH_NPY/train/*.npy')
    # train_data, valid_data = seed_MITBIH(train_files, model.hyperparameters['dataSeed'])
    train_data, valid_data = FOLD5_MITBIH(train_files, model.hyperparameters['dataSeed'])
    
    train_dataset    = MIT_DATASET(train_data,featureLength,srTarget, classes, dataNorm, model.hyperparameters['trainaug'], True)
    valid_dataset    = MIT_DATASET(valid_data,featureLength,srTarget, classes, dataNorm, False)
    test_dataset     = MIT_DATASET(test_data,featureLength,srTarget, classes, dataNorm, False)
    AMC_dataset      = MIT_DATASET(AMC_data,featureLength,srTarget, classes, dataNorm, False)
    CPSC2020_dataset = MIT_DATASET(CPSC2020_data,featureLength, srTarget, classes, dataNorm,False)
    # CU_dataset       = MIT_DATASET(CU_data,featureLength, srTarget, classes, False)
    ESC_dataset      = MIT_DATASET(ESC_data,featureLength, srTarget, classes, False)
    # FANTASIA_dataset = MIT_DATASET(FANTASIA_data,featureLength, srTarget, classes, False)
    INCART_dataset   = MIT_DATASET(INCART_data,featureLength, srTarget, classes, dataNorm, False)
    NS_dataset       = MIT_DATASET(NS_data,featureLength, srTarget, classes, dataNorm, False)
    # STDB_dataset     = MIT_DATASET(STDB_data,featureLength, srTarget, classes, dataNorm, False)
    SVDB_dataset     = MIT_DATASET(SVDB_data,featureLength, srTarget, classes, dataNorm, False)
    # AMCREAL_dataset  = MIT_DATASET(AMCREAL_data,featureLength, srTarget, classes, dataNorm, False)

    if model.hyperparameters['sampler']:
        train_loader = DataLoader(train_dataset, batch_size = model.hyperparameters['batch_size'], shuffle = False, num_workers=NUM_WORKERS//4, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset), drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size = model.hyperparameters['batch_size']//4, shuffle = False, num_workers=NUM_WORKERS//4, pin_memory=True, sampler=ImbalancedDatasetSampler(valid_dataset), drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size = model.hyperparameters['batch_size'], shuffle = True, num_workers=NUM_WORKERS//4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size = model.hyperparameters['batch_size']//4, shuffle = False, num_workers=NUM_WORKERS//4, pin_memory=True)

    batch_size = 128
    test_loader     = DataLoader(test_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    AMC_loader      = DataLoader(AMC_dataset,batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    CPSC2020_loader = DataLoader(CPSC2020_dataset,batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    # CU_loader       = DataLoader(CU_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    ESC_loader      = DataLoader(ESC_dataset,batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    # FANTASIA_loader = DataLoader(FANTASIA_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    INCART_loader   = DataLoader(INCART_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    NS_loader       = DataLoader(NS_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    # STDB_loader     = DataLoader(STDB_dataset, batch_size = batch_size, num_workers=2, shuffle = False)
    SVDB_loader     = DataLoader(SVDB_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)

    
    wandb_logger = pl_loggers.WandbLogger(save_dir=f"{wandb.config.path_logRoot}/{model.experiment_name}", name=model.experiment_name, project=wandb.config.project, offline=False)

    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch',)
    early_stop_callback = EarlyStopping(monitor='val_loss', mode="min", patience=12, verbose=False)
    loss_checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=f"{wandb.config.path_logRoot}/{model.experiment_name}/weight/", filename="best_val_loss", save_top_k=1, verbose=False)
    # metric_checkpoint_callback = ModelCheckpoint(monitor='val_AUPRC_Class1Raw', mode='max', dirpath=f"{wandb.config.path_logRoot}/{model.experiment_name}/weight/", filename="best_val_metric", save_top_k=1, verbose=False)

    trainer = pl.Trainer(accumulate_grad_batches=2,
                        gradient_clip_val=0.1,
                        accelerator='gpu',
                        devices=-1,
                        strategy ='dp',
                        max_epochs=500, # 80
                        sync_batchnorm=True,
                        benchmark=False,
                        deterministic=True,
                        check_val_every_n_epoch=5,
                        # callbacks=[loss_checkpoint_callback, metric_checkpoint_callback, lr_monitor_callback, early_stop_callback],# StochasticWeightAveraging(swa_lrs=0.05)], #
                        callbacks=[loss_checkpoint_callback, lr_monitor_callback, early_stop_callback, pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.6, swa_lrs=1e-5)], #
                        logger = wandb_logger,
                        precision= 32 # 'bf16', 16, 32
    )
    
    set_seed()
    trainer.fit(model, train_loader, valid_loader)
    
    set_seed()
    result_test = trainer.test(model, test_loader, ckpt_path='best')
    result_AMC = trainer.test(model, AMC_loader, ckpt_path='best')
    result_CPSC2020 = trainer.test(model, CPSC2020_loader, ckpt_path='best')
    # result_CU = trainer.test(model, CU_loader,ckpt_path='best')
    result_ESC = trainer.test(model, ESC_loader, ckpt_path='best')
    # result_FANTASIA = trainer.test(model, FANTASIA_loader,ckpt_path='best')
    result_INCART = trainer.test(model, INCART_loader, ckpt_path='best')
    result_NS = trainer.test(model, NS_loader, ckpt_path='best')
    # result_STDB = trainer.test(model, STDB_loader,ckpt_path='best')
    result_SVDB = trainer.test(model, SVDB_loader, ckpt_path='best')

def test(path, testPlot=False):
    set_seed()
    model = PVC_NET.load_from_checkpoint(path)

    classes = model.hyperparameters['outChannels']
    srTarget = model.hyperparameters['srTarget']
    featureLength = model.hyperparameters['featureLength']    
    dataNorm = model.hyperparameters['dataNorm']
    
    test_dataset = MIT_DATASET(test_data,featureLength,srTarget, classes, dataNorm, False)
    AMC_dataset = MIT_DATASET(AMC_data,featureLength,srTarget, classes, dataNorm, False)
    CPSC2020_dataset = MIT_DATASET(CPSC2020_data,featureLength, srTarget, classes, dataNorm,False)
    # CU_dataset = MIT_DATASET(CU_data,featureLength, srTarget, classes, False)
    ESC_dataset = MIT_DATASET(ESC_data,featureLength, srTarget, classes, False)
    # FANTASIA_dataset = MIT_DATASET(FANTASIA_data,featureLength, srTarget, classes, False)
    INCART_dataset = MIT_DATASET(INCART_data,featureLength, srTarget, classes, dataNorm, False)
    NS_dataset = MIT_DATASET(NS_data,featureLength, srTarget, classes, dataNorm, False)
    STDB_dataset = MIT_DATASET(STDB_data,featureLength, srTarget, classes, dataNorm, False)
    SVDB_dataset = MIT_DATASET(SVDB_data,featureLength, srTarget, classes, dataNorm, False)
    # AMCREAL_dataset = MIT_DATASET(AMCREAL_data,featureLength, srTarget, classes, dataNorm, False)

    batch_size = 128
    test_loader     = DataLoader(test_dataset, batch_size = batch_size, num_workers=2, shuffle = False)
    AMC_loader      = DataLoader(AMC_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    CPSC2020_loader = DataLoader(CPSC2020_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    # CU_loader       = DataLoader(CU_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    ESC_loader      = DataLoader(ESC_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    # FANTASIA_loader = DataLoader(FANTASIA_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    INCART_loader   = DataLoader(INCART_dataset, batch_size = batch_size, num_workers=2, shuffle = False)
    NS_loader       = DataLoader(NS_dataset, batch_size = batch_size, num_workers=2, shuffle = False)
    # STDB_loader     = DataLoader(STDB_dataset, batch_size = batch_size, num_workers=2, shuffle = False)
    SVDB_loader     = DataLoader(SVDB_dataset, batch_size = batch_size, num_workers=2, shuffle = False)
    
    trainer = pl.Trainer(accumulate_grad_batches=8,
                        gradient_clip_val=0.1,
                        accelerator='gpu',
                        devices=-1,
                        strategy ='dp',
                        max_epochs=200, # 80
                        sync_batchnorm=True,
                        benchmark=False,
                        deterministic=True,
                        check_val_every_n_epoch=10,
                        # callbacks=[loss_checkpoint_callback, lr_monitor_callback, early_stop_callback],# , StochasticWeightAveraging(swa_lrs=0.0001)], #
                        # logger = wandb_logger,
                        precision= 32 # 'bf16', 16, 32
    )
    
    model.testPlot=testPlot
    
    result_test = trainer.test(model, test_loader)
    result_AMC = trainer.test(model, AMC_loader)
    result_CPSC2020 = trainer.test(model, CPSC2020_loader)
    # result_CU = trainer.test(model, CU_loader)
    result_ESC = trainer.test(model, ESC_loader)
    # result_FANTASIA = trainer.test(model, FANTASIA_loader)
    result_INCART = trainer.test(model, INCART_loader)        
    result_NS = trainer.test(model, NS_loader)
    # result_STDB = trainer.test(model, STDB_loader)
    result_SVDB = trainer.test(model, SVDB_loader)


class PVC_NET(pl.LightningModule):
    def __init__(self,hyperparameters):
        super(PVC_NET, self).__init__()
        
        self.hyperparameters = hyperparameters
        self.experiment_name = str(self.hyperparameters).replace("{","").replace("}","").replace("'","").replace(": ","").replace(", ","_").split('_project')[0] # cut name as it is too long
        path = f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/"
        print(f'saving path : {path}')
        os.makedirs(path, mode=0o777, exist_ok=True)
        
        self.srTarget = self.hyperparameters['srTarget']
        self.featureLength= self.hyperparameters['featureLength']
        self.thresholdRPeak = self.hyperparameters['thresholdRPeak'] # 0.7
        self.learning_rate = self.hyperparameters['learning_rate']
        self.dataNorm =self.hyperparameters['dataNorm']
        self.youden_index = 0.25
        self.testPlot = False
        
        if self.hyperparameters['norm'] =='layer':
            norm = ('group',{'num_groups':1})
        elif self.hyperparameters['norm'] =='group':
            norm = ('group',{'num_groups':8})
        else:
            norm = self.hyperparameters['norm']
        
        # define model using hyperparamters
        if 'efficient' in hyperparameters['modelName'] or 'resnet' in hyperparameters['modelName']:
            self.net = nets.UNet(modelName = hyperparameters['modelName'], 
                            spatial_dims = hyperparameters['spatial_dims'],
                            in_channels = hyperparameters['inChannels'],
                            out_channels = hyperparameters['outChannels'],
                            norm = norm,
                            upsample = hyperparameters['upsample'],
                            dropout = hyperparameters['dropout'],
                            supervision = hyperparameters['supervision'],
                            skipModule = hyperparameters['skipModule'],
                            skipASPP =  hyperparameters['skipASPP'],
                            segheadModule = hyperparameters['segheadModule'],
                            se_module= hyperparameters['se'],
                            mtl=hyperparameters['mtl'],
                           )
            
#         elif 'U2' in hyperparameters['modelName']:
#             self.net = nets.U2NET(in_ch=hyperparameters['in_channels'],
#                                   out_ch=hyperparameters['out_channels'],
#                                   nnblock = hyperparameters['nnblock'],
#                                   ASPP = hyperparameters['ASPP'],
#                                   FFC = hyperparameters['FFC'],
#                                   acm = hyperparameters['acm'],
#                                   dropout = hyperparameters['dropout'],
#                                   temperature=1,
#                                   norm = hyperparameters['norm'],
#                                  )
            
#         elif 'unetr' in hyperparameters['modelName']:
#             self.net= monai.networks.nets.UNETR(hyperparameters['in_channels'], 
#                                                 hyperparameters['out_channels'],
#                                                 2048,
#                                                 feature_size = 16,
#                                                 hidden_size = 768,
#                                                 mlp_dim = 3072,
#                                                 num_heads = 12,
#                                                 pos_embed = 'conv',
#                                                 norm_name= hyperparameters['norm'],
#                                                 conv_block = True,
#                                                 res_block = True,
#                                                 dropout_rate = 0.0,
#                                                 spatial_dims = hyperparameters['spatial_dims'],)
            
        # define loss using hyperparameters
        if hyperparameters['lossFn']=='BCE':
            self.lossFn = nn.BCELoss()
        elif hyperparameters['lossFn']=='FOCAL':
            self.lossFn = monai.losses.FocalLoss(include_background=True, gamma=2, reduction='none')
        elif hyperparameters['lossFn']=='BCEFOCAL':
            self.lossFn = BCEFocalLoss(alpha=1, gamma=2)
        # elif hyperparameters['lossFn']=='DICEBCE':
        #     self.lossFn = DiceBCE()
        # elif hyperparameters['lossFn']=='DICEFOCAL':
        #     self.lossFn = monai.losses.DiceFocalLoss()
        # elif hyperparameters['lossFn']=='PROPOTIONAL':
        #     self.lossFn = PropotionalLoss(per_image=False, smooth=1e-7, beta=0.7, bce=True)

        self.save_hyperparameters()
        
    def compute_loss(self, yhat, y):
        if isinstance(yhat,list) or isinstance(yhat,tuple):
            # yhat, loss_dp = yhat # ACM loss
            # loss = self.lossFn(yhat,y) # ACM loss
            # loss = loss + loss_dp # ACM loss
            yhat, yhat_mtl = yhat
            loss = self.lossFn(yhat,y) + self.lossFn(yhat_mtl.squeeze(), F.adaptive_max_pool1d(y,1)[:,1].squeeze())
            # loss = F.binary_cross_entropy(yhat,y) + F.binary_cross_entropy(yhat_mtl, F.adaptive_max_pool1d(y,1))
            
        else:
            loss = self.lossFn(yhat, y)
        return loss
    
    def forward(self, x):
        result = self.net(x)
        return result
        
    def sliding_window_inference(self, x): # Inference Sliding window using MONAI API: Using this only valid and test when size of input is larger than 2048
        def predictor(x, return_idx = 0): # in case of network gets multiple output, we will use only 1st output
            result = self.forward(x)
            if isinstance(result, list) or isinstance(result, tuple):
                return result[return_idx]
            else:
                return result        
        return sliding_window_inference(x, self.featureLength, 8, predictor, mode='gaussian', overlap=0.75)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, pct_start=0.02, total_steps=self.trainer.estimated_stepping_batches)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
    
    def pipeline(self, batch, batch_idx):
        
        x = batch['signal'].float()
        y = batch['y_seg'].float()
        
        dataSource = batch['dataSource'][0]
        
        fname = batch['fname']
        pid = batch['pid']
        time = batch['time']
        
        yhat = self.sliding_window_inference(x) if x.shape[-1] > self.featureLength else self.forward(x)
        loss = self.compute_loss(yhat, y)

        if isinstance(yhat,tuple) or isinstance(yhat,list): # in case multi output model such as U2NET while training
            yhat = yhat[0]
                
        return {'loss':loss, "x": x, "y": y, "yhat":yhat, "dataSource":dataSource,'fname':fname}
    
    def training_step(self, batch, batch_idx):        
        result = self.pipeline(batch, batch_idx)
        self.log('loss', result['loss'], on_step=False, on_epoch=True, prog_bar=True)
        return {"loss":result['loss'], "x": result['x'], "y": result['y'], "yhat":result['yhat'], "dataSource":result['dataSource'], 'fname':result['fname']}

    def validation_step(self, batch, batch_idx):
        result = self.pipeline(batch, batch_idx)
        self.log('val_loss', result['loss'], on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss":result['loss'], "x": result['x'], "y": result['y'], "yhat":result['yhat'], "dataSource":result['dataSource'], 'fname':result['fname']}
    
    def test_step(self, batch, batch_idx):
        result = self.pipeline(batch, batch_idx)
        self.log('test_loss', result['loss'], on_step=False, on_epoch=True)
        return {"test_loss":result['loss'], "x": result['x'], "y": result['y'], "yhat":result['yhat'], "dataSource":result['dataSource'], 'fname':result['fname']}
    
    def validation_epoch_end(self, outputs):
        
        self.fnames = []
        self.evaluations(outputs,'val', False)

    def test_epoch_end(self, outputs):        
        
        self.fnames = []
        if outputs[0]['dataSource'][0]==3:
            self.dataSource = 'testMIT'
            data = test_data
        elif outputs[0]['dataSource'][0]==11:
            self.dataSource = 'testAMC'
            data = AMC_data
        elif outputs[0]['dataSource'][0]==12:
            self.dataSource = 'testCPSC2020'
            data = CPSC2020_data
        elif outputs[0]['dataSource'][0]==13:
            self.dataSource = 'testCU'
            data = CU_data
        elif outputs[0]['dataSource'][0]==14:
            self.dataSource = 'testESC'
            data = ESC_data
        elif outputs[0]['dataSource'][0]==15:
            self.dataSource = 'testFANTASIA'
            data = FANTASIA_data
        elif outputs[0]['dataSource'][0]==16:
            self.dataSource = 'testINCART'
            data = INCART_data
        elif outputs[0]['dataSource'][0]==17:
            self.dataSource = 'testNS'
            data = NS_data
        elif outputs[0]['dataSource'][0]==18:
            self.dataSource = 'testSTDB'
            data = STDB_data
        elif outputs[0]['dataSource'][0]==19:
            self.dataSource = 'testSVDB'
            data = SVDB_data
        elif outputs[0]['dataSource'][0]==20:
            self.dataSource = 'testAMCREAL'
            data = AMCREAL_data
        
        for d in data:
            self.fnames.append(f"{d['pid']}_{d['time']}")

        self.evaluations(outputs, self.dataSource, True)
    
    def apply_threshold(self, pred, t):
        try:
            result = pred.clone()
        except:
            result = pred.copy()
        result[result>=t]= 1
        result[result<t]= 0
        return result
    
    def evaluations(self, outputs, dataSource, plot=False):
        
        # fnames = []
        xs = []
        ys= []
        yhatsRaw = []                    
        yhatsRefined = []                    
        
        ysClass1Raw = []
        ysClass2Raw = []
        ysClass3Raw = []
        
        yhatsClass1Raw = []
        yhatsClass2Raw = []
        yhatsClass3Raw = []
        
        ysClass1Refined = []
        ysClass2Refined = []
        ysClass3Refined = []
        
        yhatsClass1Refined = []
        yhatsClass2Refined = []
        yhatsClass3Refined = []

        RRaw_TP = 0
        RRaw_FP = 0
        RRaw_FN = 0
        RRefined_TP = 0
        RRefined_FP = 0
        RRefined_FN = 0
        
        for output in outputs:
            # fnames.extend(output['fname'])
            xs.extend(output["x"].cpu().detach().numpy())
            ys.extend(output["y"].cpu().detach().numpy())

            for i in range(len(output["y"])):
                y = output["y"][i].cpu().detach().numpy()
                yhatRaw = output["yhat"][i].cpu().detach().numpy()
                yhatRefined = self.postProcessByRPeak(output["yhat"][i].cpu().detach().numpy())
                
                yhatsRaw.append(yhatRaw)
                yhatsRefined.append(yhatRefined)
                
                yhatRaw_eval = self.eval_Peak(yhatRaw, y)                
                RRaw_TP += yhatRaw_eval['R_TPs']
                RRaw_FP += yhatRaw_eval['R_FPs']
                RRaw_FN += yhatRaw_eval['R_FNs']
                ysClass1Raw.extend(yhatRaw_eval['ys_class1'])
                yhatsClass1Raw.extend(yhatRaw_eval['yhats_class1'])                
                ysClass2Raw.extend(yhatRaw_eval['ys_class2'])
                yhatsClass2Raw.extend(yhatRaw_eval['yhats_class2'])                
                ysClass3Raw.extend(yhatRaw_eval['ys_class3'])
                yhatsClass3Raw.extend(yhatRaw_eval['yhats_class3'])                

                yhatRefined_eval = self.eval_Peak(yhatRefined, y)                
                RRefined_TP += yhatRefined_eval['R_TPs']
                RRefined_FP += yhatRefined_eval['R_FPs']
                RRefined_FN += yhatRefined_eval['R_FNs']
                ysClass1Refined.extend(yhatRefined_eval['ys_class1'])
                yhatsClass1Refined.extend(yhatRefined_eval['yhats_class1'])
                ysClass2Refined.extend(yhatRefined_eval['ys_class2'])
                yhatsClass2Refined.extend(yhatRefined_eval['yhats_class2'])
                ysClass3Refined.extend(yhatRefined_eval['ys_class3'])
                yhatsClass3Refined.extend(yhatRefined_eval['yhats_class3'])
                
        del output
        
        fnames = np.array(self.fnames)
        xs = np.array(xs)
        ys = np.array(ys)
        yhatsRaw = np.array(yhatsRaw)
        yhatsRefined = np.array(yhatsRefined)
        
        ysClass1Raw = np.array(ysClass1Raw)
        ysClass2Raw = np.array(ysClass2Raw)
        ysClass3Raw = np.array(ysClass3Raw)
        ysClass1Refined = np.array(ysClass1Refined)
        ysClass2Refined = np.array(ysClass2Refined)
        ysClass3Refined = np.array(ysClass3Refined)
        yhatsClass1Raw = np.array(yhatsClass1Raw)
        yhatsClass2Raw = np.array(yhatsClass2Raw)
        yhatsClass3Raw = np.array(yhatsClass3Raw)
        yhatsClass1Refined = np.array(yhatsClass1Refined)
        yhatsClass2Refined = np.array(yhatsClass2Refined)
        yhatsClass3Refined = np.array(yhatsClass3Refined)
        
        def eval_cm(ys, yhats, name):
            """
            input shape shoud be only [ B ]
            """
            
            TP = 0
            FN = 0
            FP = 0
            TN = 0

            auc = sklearn.metrics.roc_auc_score(ys, yhats)
            ap  = sklearn.metrics.average_precision_score(ys, yhats)
            # self.youden_index = find_maxF1(ys, yhats)
            
            negativeIdx = np.where(ys == 0)
            positiveIdx = np.where(ys != 0)
            
            for i in range(len(negativeIdx[0])):
                z = negativeIdx[0][i]
                FP = FP+1 if yhats[z]>=self.youden_index else FP
                TN = TN+1 if yhats[z]<self.youden_index else TN
            for i in range(len(positiveIdx[0])):
                n = positiveIdx[0][i]
                TP = TP+1 if yhats[n]>=self.youden_index else TP
                FN = FN+1 if yhats[n]<self.youden_index else FN
            
            # print(TP,TN,FP,FN)
            sen = TP/(TP+FN)
            spe = TN/(TN+FP)
            acc = (TP+TN)/(TP+FN+FP+TN)
            bacc = (sen+spe)/2
            
            # f1 = sklearn.metrics.f1_score(gt1, apply_threshold(yhat1,yi))
            # pr  = sklearn.metrics.precision_score(gt1,apply_threshold(yhat1,yi))
            f1 = 2*TP/(2*TP+FP+FN)
            ppv = TP/(TP+FP)
            npv = TN/(TN+FN)

            # plot performance
            plt.figure(figsize=(20,4))
            plt.subplot(131)           
            plt.title(f'Histogram of likelihood (Youden index : {self.youden_index:.3f})')
            plt.xlabel('Likelihood')
            plt.ylabel('Normalized samples')
            plt.hist(yhats[negativeIdx],bins=50,density=True,label='Likelihood of Negative Cases',alpha=0.5)
            plt.hist(yhats[positiveIdx],bins=50,density=True,label='Likelihood of Positive Cases',alpha=0.5)
            plt.vlines(self.youden_index,0,10,label='Youden index',color='r',alpha=0.5)
            plt.xlim([0,1])
            plt.legend()
            
            plt.subplot(132)
            plt.title(f'ROC (AUROC : {auc:.3f})')
            plt.xlabel('1-Specificity')
            plt.ylabel('Sensitivity')
            fpr, tpr, _ = roc_curve(ys, yhats)
            plt.plot(fpr, tpr)
            
            plt.subplot(133)        
            plt.title(f'Precision-Recall (AP : {ap:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            prec, recall, _ = precision_recall_curve(ys, yhats)
            plt.plot(recall, prec)
            # plt.show()
            os.makedirs(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/", mode=0o777, exist_ok=True)
            plt.savefig(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/performance_{dataSource}_{name}.png")
            plt.close()
            
            return {"TP":TP,"TN":TN,"FN":FN,"FP":FP,
                    "auc":auc,"ap":ap,"sen":sen,"spe":spe,"acc":acc,"bacc":bacc,"f1":f1,"ppv":ppv,"npv":npv,"youdenIndex":self.youden_index}
        
        try:
            cmClass1Raw = eval_cm(ysClass1Raw, yhatsClass1Raw,'Class1Raw')
            cmClass1Refined = eval_cm(ysClass1Refined, yhatsClass1Refined,'Class1Refined')
            cmClass2Raw = eval_cm(ysClass2Raw, yhatsClass2Raw,'Class2Raw')
            cmClass2Refined = eval_cm(ysClass2Refined, yhatsClass2Refined,'Class2Refined')
            cmClass3Raw = eval_cm(ysClass3Raw, yhatsClass3Raw,'Class3Raw')
            cmClass3Refined = eval_cm(ysClass3Refined, yhatsClass3Refined,'Class3Refined')
        except:
            pass
        
        RRaw_sen = RRaw_TP/(RRaw_TP+RRaw_FN)
        RRaw_pp  = RRaw_TP/(RRaw_TP+RRaw_FP)
        RRaw_err = (RRaw_FP+RRaw_FN)/(RRaw_TP+RRaw_FP+RRaw_FN)

        RRefined_sen = RRefined_TP/(RRefined_TP+RRefined_FN)
        RRefined_pp  = RRefined_TP/(RRefined_TP+RRefined_FP)
        RRefined_err = (RRefined_FP+RRefined_FN)/(RRefined_TP+RRefined_FP+RRefined_FN)
            
        def logcmResult(cmPVC, refined):
            
            self.log(f'{dataSource}_TP_{refined}',cmPVC['TP'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_FN_{refined}',cmPVC['FN'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_FP_{refined}',cmPVC['FP'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_TN_{refined}',cmPVC['TN'],on_step=False,on_epoch=True)

            self.log(f'{dataSource}_AUPRC_{refined}',cmPVC['ap'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_AUROC_{refined}',cmPVC['auc'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_ACC_{refined}',cmPVC['acc'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_SEN_{refined}',cmPVC['sen'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_SPE_{refined}',cmPVC['spe'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_BACC_{refined}',cmPVC['bacc'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_F1_{refined}',cmPVC['f1'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_PPV_{refined}',cmPVC['ppv'],on_step=False,on_epoch=True)
            self.log(f'{dataSource}_NPV_{refined}',cmPVC['npv'],on_step=False,on_epoch=True)
            
            self.log(f'{dataSource}_YoudenIndex_{refined}',cmPVC['youdenIndex'],on_step=False,on_epoch=True)
            
        def logRResult(R_sen,R_ppv,R_der, refined):
            self.log(f'{dataSource}_R-DER_{refined}',R_der,on_step=False,on_epoch=True)
            self.log(f'{dataSource}_R-PPV_{refined}',R_ppv,on_step=False,on_epoch=True)
            self.log(f'{dataSource}_R-SEN_{refined}',R_sen,on_step=False,on_epoch=True)

        try:
            logcmResult(cmClass1Raw,'Class1Raw')
            logcmResult(cmClass1Refined,'Class1Refined')
        except:
            pass
        try:
            logcmResult(cmClass2Raw,'Class2Raw')
            logcmResult(cmClass2Refined,'Class2Refined')
        except:
            pass
        try:
            logcmResult(cmClass3Raw,'Class3Raw')
            logcmResult(cmClass3Refined,'Class3Refined')
        except:
            pass
        
        logRResult(RRaw_sen,RRaw_pp,RRaw_err, 'Raw')
        logRResult(RRefined_sen,RRefined_pp,RRefined_err,'Refined')
        
        os.makedirs(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/", mode=0o777, exist_ok=True)

        if len(ysClass3Raw)!=0 and len(ysClass2Raw)!=0:
            df = pd.DataFrame([ysClass1Raw, yhatsClass1Raw, ysClass2Raw, yhatsClass2Raw, ysClass3Raw, yhatsClass3Raw],
                              ['ysClass1Raw','yhatsClass1Raw','ysClass2Raw','yhatsClass2Raw','ysClass3Raw','yhatsClass3Raw'])
            df = df.T
            df.to_csv(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/likelihood_{dataSource}_Raw.csv",index=False)

            df = pd.DataFrame([ysClass1Refined, yhatsClass1Refined,ysClass2Refined, yhatsClass2Refined,ysClass3Refined, yhatsClass3Refined],
                              ['ysClass1Refined','yhatsClass1Refined','ysClass2Refined','yhatsClass2Refined','ysClass3Refined','yhatsClass3Refined'])
            df = df.T
            df.to_csv(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/likelihood_{dataSource}_Refined.csv",index=False)
        elif len(ysClass2Raw)!=0:
            df = pd.DataFrame([ysClass1Raw, yhatsClass1Raw, ysClass2Raw, yhatsClass2Raw],['ysClass1Raw','yhatsClass1Raw','ysClass2Raw','yhatsClass2Raw'])
            df = df.T
            df.to_csv(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/likelihood_{dataSource}_Raw.csv",index=False)

            df = pd.DataFrame([ysClass1Refined, yhatsClass1Refined,ysClass2Refined, yhatsClass2Refined],
                              ['ysClass1Refined','yhatsClass1Refined','ysClass2Refined','yhatsClass2Refined'])
            df = df.T
            df.to_csv(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/likelihood_{dataSource}_Refined.csv",index=False)
        else:
            df = pd.DataFrame([ysClass1Raw, yhatsClass1Raw],['ysClass1Raw','yhatsClass1Raw'])
            df = df.T
            df.to_csv(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/likelihood_{dataSource}_Raw.csv",index=False)

            df = pd.DataFrame([ysClass1Refined, yhatsClass1Refined],['ysClass1Refined','yhatsClass1Refined'])
            df = df.T
            df.to_csv(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/metric/likelihood_{dataSource}_Refined.csv",index=False)
                
        try:
            for i in range(len(yhatsRefined)):
                yhatsRefined[i] = self.postProcessByYoudenIndex(yhatsRefined[i], cmClass1Refined['youdenIndex'])
            self.plotCaseResult(xs, ys, yhatsRaw, yhatsRefined, fnames) if self.testPlot else 0
        except:
            pass
        
        
    def eval_Peak(self, yhat, y):
        
        """
        input y: CxSignal
        input yhat : CXSignal
        """
        classes = yhat.shape[0]-1

        try:
            yhat_ = yhat.clone()
        except:
            yhat_ = yhat.copy()
            
        yhat_[0] = self.apply_threshold(yhat_[0],self.thresholdRPeak)
            
        ys_class1 = []
        ys_class2 = []
        ys_class3 = []
        yhats_class1 = []
        yhats_class2 = []
        yhats_class3 = []

        # evalutation of R-peak
        R_TP = []
        R_FP = []
        R_FN = []
        
        result_y, count_y = label(y[0])

        for j in range(1, count_y+1):
            index = np.where(result_y == j)[0]
            start = index[0]
            end = index[-1]
            
            try:
                yhat0_mean = torch.nanmean(yhat_[0,start:end+1])
                yhat1_mean = torch.nanmean(yhat_[1,start:end+1])
                yhat2_mean = torch.nanmean(yhat_[2,start:end+1])
                yhat3_mean = torch.nanmean(yhat_[3,start:end+1])
            except:
                try:
                    yhat0_mean = np.nanmean(yhat_[0,start:end+1])
                    yhat1_mean = np.nanmean(yhat_[1,start:end+1])
                    yhat2_mean = np.nanmean(yhat_[2,start:end+1])
                    yhat3_mean = np.nanmean(yhat_[3,start:end+1])
                except:
                    pass
                
            # evalutation of R-peak : TP, FN
            if 1 in y[0,start:end+1] and yhat0_mean>=self.thresholdRPeak:
                R_TP.append(1)
            elif 1 in y[0,start:end+1] and yhat0_mean<self.thresholdRPeak:
                R_FN.append(1)            
                
            # evalutation of PVC : just return likelihood
            if 0 in y[1,start:end]:
                ys_class1.append(0)
                yhats_class1.append(yhat1_mean)
            elif 1 in y[1,start:end]:
                ys_class1.append(1)
                yhats_class1.append(yhat1_mean)
                
            try:
                if 0 in y[2,start:end]:
                    ys_class2.append(0)
                    yhats_class2.append(yhat2_mean)
                elif 1 in y[2,start:end]:
                    ys_class2.append(1)
                    yhats_class2.append(yhat2_mean)

                if 0 in y[3,start:end]:
                    ys_class3.append(0)
                    yhats_class3.append(yhat3_mean)
                elif 1 in y[3,start:end]:
                    ys_class3.append(1)
                    yhats_class3.append(yhat3_mean)
            except:
                pass
            
        # print('B',count_y,np.array(ys).shape, np.array(yhats).shape, len(R_TP),len(R_FP),len(R_FN))
        result_yhat, count_yhat = label(yhat_[0])

        for j in range(1,count_yhat+1):
            index = np.where(result_yhat == j)[0]
            start = index[0]
            end = index[-1]
            
            try:
                yhat0_mean = torch.nanmean(yhat_[0,start:end+1])
                yhat1_mean = torch.nanmean(yhat_[1,start:end+1])
                yhat2_mean = torch.nanmean(yhat_[2,start:end+1])
                yhat3_mean = torch.nanmean(yhat_[3,start:end+1])
            except:                
                try:
                    yhat0_mean = np.nanmean(yhat_[0,start:end+1])
                    yhat1_mean = np.nanmean(yhat_[1,start:end+1])
                    yhat2_mean = np.nanmean(yhat_[2,start:end+1])
                    yhat3_mean = np.nanmean(yhat_[3,start:end+1])
                except:
                    pass
                

            # evalutation of R-peak : FP
            if 1 not in y[0,start:end+1]:
                R_FP.append(1)

            # evalutation of PVC : FP
            if 1 not in y[1,start:end+1] and 1 in yhat_[1,start:end+1]:
                ys_class1.append(0)
                yhats_class1.append(yhat1_mean)
            
            try:
                if 1 not in y[2,start:end+1] and 1 in yhat_[2,start:end+1]:
                    ys_class2.append(0)
                    yhats_class2.append(yhat2_mean)
                    
                if 1 not in y[3,start:end+1] and 1 in yhat_[3,start:end+1]:
                    ys_class3.append(0)
                    yhats_class3.append(yhat3_mean)
            except:
                pass
        
        return {
                'R_TPs':np.sum(R_TP), 'R_FNs':np.sum(R_FN), 'R_FPs':np.sum(R_FP),
                'ys_class1':np.array(ys_class1), 'yhats_class1':np.array(yhats_class1),
                'ys_class2':np.array(ys_class2), 'yhats_class2':np.array(yhats_class2),
                'ys_class3':np.array(ys_class3), 'yhats_class3':np.array(yhats_class3)
               }
    
    def postProcessByRPeak(self, yhat):
        """
        input : yhat [C x S]
        output : yhat [C x S]
        
        Rule 0. R-peak는 self.thresholdRPeak로 binarize한다.
        Rule 1. R-peak의 간격이 특정 간격보다 작으면 무시한다. 
        Rule 2. R-peak가 아니면 PVC, AFIB도 아니다. # R-peak에 살짝 마진을 주고 PVC, AFIB과 곱해준다
        threshold = int(srTarget*.05)
        """

        yhat_ = yhat.copy()
        # Rule 0.
        yhat_[0] = self.apply_threshold(yhat_[0], self.thresholdRPeak)
        
        # Rule 1. fill in and remove R- peak
        threshold_hole = int(self.srTarget*.2*0.2) # 20% of R-peak 
        yhat_[0] = morphology.remove_small_holes(yhat_[0].astype(bool), threshold_hole).astype(float) # fill in
        
        threshold_object = int(self.srTarget*.2*0.4) # 40% of R-peak seg
        yhat_[0] = morphology.remove_small_objects(yhat_[0].astype(bool), threshold_object).astype(float) # remove small R-peak
        yhat_0_dilated = ndimage.binary_dilation(yhat_[0],iterations=int(self.srTarget*.1*0.1))         #dilated
        
        # Rule 2.
        yhat_[1] = yhat_0_dilated*yhat_[1] 
        try:
            yhat_[2] = yhat_0_dilated*yhat_[2] 
            yhat_[3] = yhat_0_dilated*yhat_[3] 
        except:
            pass
        return yhat_
    
    def postProcessSeg(self, yhat):
        yhat_ = yhat.copy()
        threshold_hole = int(self.srTarget*.2*0.2) # 20% of R-peak 
        yhat_ = morphology.remove_small_holes(yhat_.astype(bool), threshold_hole).astype(float) # fill in
        
        threshold_object = int(self.srTarget*.2*0.4) # 40% of R-peak seg
        yhat_ = morphology.remove_small_objects(yhat_.astype(bool), threshold_object).astype(float) # remove small R-peak
        return yhat_
    
    def postProcessByYoudenIndex(self, yhat, threshold):
        """
        input : yhat [C x S]
        output : yhat [C x S]        
        only using yhat without y
        """
        yhat[0]= yhat[0].round()
        yhat[0] = self.postProcessSeg(yhat[0])
        result, count_yhat = label(yhat[0])
        
        for j in range(1, count_yhat+1):
            index = np.where(result == j)[0]
            start = index[0]
            end = index[-1]
            margin = int(self.srTarget*.2*.1)
            
            yhat1_mean = np.nanmean(yhat[1,start:end+1])
            
            # evalutation of PVC : FP
            if yhat1_mean >= threshold:
                yhat[1,start-margin:end+1+margin] = 1
            else:
                yhat[1,start-margin:end+1+margin] = 0
                
        yhat = yhat.round()
        return yhat
    
    def plotCaseResult(self, x, y, yhat1, yhat2, fname):
        t = np.linspace(0,x.shape[-1]/self.srTarget, x.shape[-1]) # for x-axis ticks
        
        for idx in range(len(y)):
            plt.figure(figsize=(20,12))
            plt.subplot(221)
            plt.title(f'Prediction result of {self.dataSource}')
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized ECG')
            plt.plot(t,x[idx,0],alpha=0.9,color='black',label='ECG signal')
            plt.plot(t,yhat1[idx,0],alpha=0.5,color='b',label='R Peak prediction (Likelihood)')
            plt.plot(t,yhat1[idx,1],alpha=0.7,color='r',label='PVC prediction (Likelihood)')
            # other annotation            
            
            signal_min = np.min(x[idx,0]) - .1
            signal_max = np.max(x[idx,0]) + .1

            try:
                plt.plot(t,yhat1[idx,2],alpha=0.7,color='g',label='AFIB prediction (Likelihood)')
                plt.plot(t,yhat1[idx,3],alpha=0.7,color='orange',label='Others prediction (Likelihood)')
            except:
                pass
            plt.xticks(np.arange(0, len(t)/self.srTarget, step=1))
            plt.ylim([signal_min-.1,signal_max+.1])
            plt.legend(loc=1)
            
            # plt.subplot(222)
            # plt.title(f'Raw signal of {self.dataSource}')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Normalized ECG')
            # plt.plot(t,x[idx,0],alpha=0.9,color='black',label='ECG signal')
            # plt.xticks(np.arange(0, len(t)/self.srTarget, step=1))
            # plt.ylim([0,1.5])
            # plt.legend(loc=1)
            
            plt.subplot(223)
            plt.title(f'Refined Prediction result of {self.dataSource}')
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized ECG')
            
            idx_QRS = get_Binaryindex(yhat2[idx,0])
            idx_PVC = get_Binaryindex(yhat2[idx,1])

            plt.plot(t,x[idx,0],alpha=0.9,color='black',label='ECG signal')
            # plt.plot(t,yhat2[idx,0],alpha=0.7,color='b',label='R Peak prediction (Binarized)')
            # plt.plot(t,yhat2[idx,1],alpha=0.7,color='r',label='PVC prediction (Binarized)')            
            plt.scatter(t[idx_QRS],[signal_min]*len(idx_QRS),label='R-peak',alpha=1,marker="o",color='blue')
            plt.scatter(t[idx_PVC],[signal_max]*len(idx_PVC),label='PVC',alpha=1,marker="v",color='r')
            # other annotation
            try:
                plt.plot(t,ndimage.binary_dilation(yhat2[idx,2],iterations=int(srTarget*.1*0.2)),alpha=0.7,color='g',label='AFIB prediction (Likelihood)')
                plt.plot(t,ndimage.binary_dilation(yhat2[idx,3],iterations=int(srTarget*.1*0.3)),alpha=0.7,color='orange',label='Others prediction (Likelihood)')
            except:
                pass
            plt.xticks(np.arange(0, len(t)/self.srTarget, step=1))
            plt.ylim([signal_min-.1,signal_max+.1])
            plt.legend(loc=1)

            plt.subplot(224)
            plt.title(f'Ground truth of {self.dataSource}')
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized ECG')
            
            idx_QRS = get_Binaryindex(y[idx,0])
            idx_PVC = get_Binaryindex(y[idx,1])

            plt.plot(t,x[idx,0],alpha=0.9,color='black',label='ECG signal')
#             plt.plot(t,y[idx,0],alpha=0.7,color='b',label='R Peak GT') if y is not None else y
#             plt.plot(t,y[idx,1],alpha=0.7,color='r',label='PVC GT') if y is not None else y            
            plt.scatter(t[idx_QRS],[signal_min]*len(idx_QRS),label='R-peak',alpha=1,marker="o",color='blue')
            plt.scatter(t[idx_PVC],[signal_max]*len(idx_PVC),label='PVC',alpha=1,marker="v",color='r')
            # other annotation            
            try:
                plt.plot(t,ndimage.binary_dilation(y[idx,2],iterations=int(srTarget*.1*0.2)),alpha=1,color='g',label='AFIB prediction (Likelihood)')
                plt.plot(t,ndimage.binary_dilation(y[idx,3],iterations=int(srTarget*.1*0.3)),alpha=0.7,color='orange',label='Others prediction (Likelihood)')
            except:
                pass
            plt.xticks(np.arange(0, len(t)/self.srTarget, step=1))
            plt.ylim([signal_min-.1,signal_max+.1])
            plt.legend(loc=1)
            plt.tight_layout()
            # plt.show()

            os.makedirs(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/result/", mode=0o777, exist_ok=True)
            os.makedirs(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/result/{self.dataSource}/", mode=0o777, exist_ok=True)
            plt.savefig(f"{self.hyperparameters['path_logRoot']}/{self.experiment_name}/result/{self.dataSource}/{str(fname[idx])}.png")
            plt.close()


def EDA(config_defaults):
    set_seed()
    hyperparameters = dict(config_defaults)
    model = PVC_NET(hyperparameters)

    classes = model.hyperparameters['outChannels']
    srTarget = model.hyperparameters['srTarget']
    featureLength = model.hyperparameters['featureLength']       
    dataNorm = model.hyperparameters['dataNorm']

    train_files = glob('dataset/MIT-BIH_NPY/train/*.npy')
    # train_data, valid_data = seed_MITBIH(train_files, model.hyperparameters['dataSeed'])
    train_data, valid_data = FOLD5_MITBIH(train_files, model.hyperparameters['dataSeed'])
    
    train_dataset    = MIT_DATASET(train_data,featureLength,srTarget, classes, dataNorm, model.hyperparameters['trainaug'], True)
    valid_dataset    = MIT_DATASET(valid_data,featureLength,srTarget, classes, dataNorm, False)
    test_dataset     = MIT_DATASET(test_data,featureLength,srTarget, classes, dataNorm, False)
    AMC_dataset      = MIT_DATASET(AMC_data,featureLength,srTarget, classes, dataNorm, False)
    CPSC2020_dataset = MIT_DATASET(CPSC2020_data,featureLength, srTarget, classes, dataNorm,False)
    # CU_dataset       = MIT_DATASET(CU_data,featureLength, srTarget, classes, False)
    ESC_dataset      = MIT_DATASET(ESC_data,featureLength, srTarget, classes, False)
    # FANTASIA_dataset = MIT_DATASET(FANTASIA_data,featureLength, srTarget, classes, False)
    INCART_dataset   = MIT_DATASET(INCART_data,featureLength, srTarget, classes, dataNorm, False)
    NS_dataset       = MIT_DATASET(NS_data,featureLength, srTarget, classes, dataNorm, False)
    STDB_dataset     = MIT_DATASET(STDB_data,featureLength, srTarget, classes, dataNorm, False)
    SVDB_dataset     = MIT_DATASET(SVDB_data,featureLength, srTarget, classes, dataNorm, False)
    # AMCREAL_dataset  = MIT_DATASET(AMCREAL_data,featureLength, srTarget, classes, dataNorm, False)

    batch_size = 1

    if model.hyperparameters['sampler']:
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False, num_workers=NUM_WORKERS//4, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset))
        valid_loader = DataLoader(valid_dataset, batch_size = batch_size//4, shuffle = False, num_workers=NUM_WORKERS//4, pin_memory=True, sampler=ImbalancedDatasetSampler(valid_dataset))
    else:
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=NUM_WORKERS//4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size = batch_size//4, shuffle = False, num_workers=NUM_WORKERS//4, pin_memory=True)

    p = []
    n = []
    for idx, batch in enumerate(train_loader):
        y_PVC = batch['y_PVC']
        fname = batch['fname']
        if 1 in y_PVC:
            p.append(fname)
        else:
            n.append(fname)
    print(f'check train sampler, positive segment: {len(p)} negative segment:{len(n)}')

    test_loader     = DataLoader(test_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    AMC_loader      = DataLoader(AMC_dataset,batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    CPSC2020_loader = DataLoader(CPSC2020_dataset,batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    # CU_loader       = DataLoader(CU_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    ESC_loader      = DataLoader(ESC_dataset,batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    # FANTASIA_loader = DataLoader(FANTASIA_dataset,batch_size = batch_size, num_workers=2, shuffle = False)
    INCART_loader   = DataLoader(INCART_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    NS_loader       = DataLoader(NS_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)
    # STDB_loader     = DataLoader(STDB_dataset, batch_size = batch_size, num_workers=2, shuffle = False)
    SVDB_loader     = DataLoader(SVDB_dataset, batch_size = batch_size, num_workers=NUM_WORKERS//4, shuffle = False)

    loaders = [train_loader, valid_loader, test_loader, AMC_loader, CPSC2020_loader, ESC_loader, INCART_loader, NS_loader, SVDB_loader]
    
    for l in loaders:
        batch = next(iter(l))
        signal_original = batch['signal_original']
        signal = batch['signal']
        y_seg = batch['y_seg']
        print(f"dataSource:{batch['dataSource'][0]} totalCount:{len(l)} fname:{batch['pid'][0]} shape:{signal.shape} unique:{torch.unique(signal)} ")
        
        i = 0
        idx_QRS = get_Binaryindex(y_seg[i,0].numpy())
        idx_PVC = get_Binaryindex(y_seg[i,1].numpy())
        signal_min = torch.min(signal[i,0]) - .2
        signal_max = torch.max(signal[i,0]) + .2
        
        plt.figure(figsize=(16,4))
        # plt.plot(signal_original[i,0],label='Orignal ECG')
        plt.plot(signal[i,0],label='Preprocessed ECG',color='black')
        plt.scatter(idx_QRS,[signal_min]*len(idx_QRS),label='R-peak',alpha=0.8,marker="o")
        plt.scatter(idx_PVC,[signal_max]*len(idx_PVC),label='PVC',alpha=0.8,marker="v")
        plt.legend()
        plt.show()