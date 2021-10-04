import sys
from helper_functions_R30 import load_cfg


# Loop Through Each Config Experiment
def execute_training(cfg):
    import numpy as np
    import os
    import pandas as pd
    import time
    from helper_functions_R30 import sig_stats, SigFilter, QTransform, LoadData, WandB, get_ckpt_path, avg_psd_calculate
    from pl_dataset_model_R30 import GwTestDataset, GwDataModule, GwModel, ImgAugTransform
    from sklearn.model_selection import KFold
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    import wandb
    import gc
    import timm
    from gwpy.frequencyseries import FrequencySeries

    SEED = 42  # Seed Everything
    seed_everything(SEED)

    """ Loop Through Each Config Experiment """
    N_FOLDS = cfg['N_FOLDS']
    fold = cfg['FOLDS'][0]
    PARTIAL = cfg['PARTIAL']

    """ Create Scipy HPF """
    hpf = SigFilter(btype='highpass',
                    cutoff=cfg['SPEC']['fmin'],
                    order=8,
                    tukey_alpha=cfg['WHITEN']['alpha'],
                    leakage_window_type=cfg['WHITEN']['leakage_window_type'],
                    )

    """ Create Q-Transforms """
    q_transforms = QTransform(spec=cfg['SPEC'], spec_mix=cfg['SPEC_MIX'], TTA=True).create_q_transforms()

    """ Get Timm Model Config Info """
    model_cfg = timm.create_model(cfg['MODEL']['name']).default_cfg

    """ Image Transforms """
    img_transform = ImgAugTransform(augs=cfg['AUG_IMG'], resize=cfg['RESIZE'], model_cfg=model_cfg)

    """ Load Training and Validation Data """
    start = time.time()
    train_data = LoadData(dataset_name=cfg['DATASET'],
                          data_type='train',
                          partial=PARTIAL)
    train_data.load_data()
    end = time.time()
    print(f'Time to Load Data: {end - start}')
    train_data.stats['NORM_SIG'] = cfg['NORM_SIG']

    """ Load Average PSD """
    start = time.time()
    psd_freqs, psd_avg = avg_psd_calculate(X_train=train_data.X[0:1000].copy(),
                                           y_train=train_data.y[0:1000].copy(),
                                           tukey_alpha=cfg['WHITEN']['alpha'],
                                           )
    end = time.time()
    print(f'Time to Avg Spectral Density {end - start}')

    # PSD Average as Frequency Class
    f0 = psd_freqs[0]
    df = psd_freqs[1] - psd_freqs[0]
    psds = []
    for i in range(3):
        psd = FrequencySeries(psd_avg[i], f0=f0, df=df)
        psds.append(psd)
    hpf.psd = psds

    """ Split Data into Training and Validation """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_idx = {}
    for i, (train_index, val_index) in enumerate(kf.split(np.ones(train_data.N))):
        fold_idx[i] = {'train': train_index.tolist(),
                       'val': val_index.tolist(),
                       }
    X_TRAIN = train_data.X[fold_idx[fold]['train']]
    Y_TRAIN = train_data.y[fold_idx[fold]['train']]
    X_VAL = train_data.X[fold_idx[fold]['val']]
    Y_VAL = train_data.y[fold_idx[fold]['val']]
    del train_data.X, train_data.y

    """ Measure Signal Statistics """
    hpf = sig_stats(X_TRAIN[0:1_000].copy(), Y_TRAIN[0:1_000], hpf)

    """ Weight and Biases Logger """
    wab = WandB(cfg)

    """ Data Module """
    dm = GwDataModule(x_train=X_TRAIN,
                      y_train=Y_TRAIN,
                      x_val=X_VAL,
                      y_val=Y_VAL,
                      batch_size=cfg['BATCH_SIZE'],
                      q_transform=q_transforms,
                      img_transform=img_transform,
                      hpf=hpf,
                      norms=train_data.stats,
                      spec_mix=cfg['SPEC_MIX'],
                      )

    """ Learning Rate Monitor """
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    """ Define Checkpoint Callback """
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['MONITOR']['metric'],
        mode=cfg['MONITOR']['mode'],
        save_last=True,
    )

    """ Load Checkpoint for Training or Start New Model """
    # Load a previously trained checkpoint
    if cfg['TRAINING_CONTINUE']['apply']:
        ckpt_path = get_ckpt_path(cfg["TRAINING_CONTINUE"]["run"],
                                  cfg["TRAINING_CONTINUE"]["run"],
                                  cfg['CKPT'],
                                  existing=True)
        gw_model = GwModel.load_from_checkpoint(ckpt_path['path'],
                                                lr_inputs=cfg['LEARNING_SCHEDULER'],
                                                aug=cfg['AUG'])
        print(f'Conitnue Training Model: {cfg["TRAINING_CONTINUE"]["run"]}')
    # Load a new model
    else:
        print(f'Training a New Model')
        # Model
        gw_model = GwModel(model_inputs=cfg['MODEL'],
                           lr_inputs=cfg['LEARNING_SCHEDULER'],
                           batch_size=cfg['BATCH_SIZE'],
                           loss_fn=cfg['LOSS'],
                           )

    """ Train the Model """
    if cfg['LIMIT_BATCHES']['apply']:
        trainer = pl.Trainer(gpus=1,
                             check_val_every_n_epoch=1,
                             max_epochs=cfg['EPOCHS'],
                             logger=[wab.wb_logger],
                             deterministic=True,
                             callbacks=[checkpoint_callback, lr_monitor],
                             limit_train_batches=cfg['LIMIT_BATCHES']['percent'],
                             limit_val_batches=cfg['LIMIT_BATCHES']['percent'],
                             precision=cfg['PRECISION'],
                             )
    else:
        trainer = pl.Trainer(gpus=1,
                             check_val_every_n_epoch=1,
                             max_epochs=cfg['EPOCHS'],
                             logger=[wab.wb_logger],
                             deterministic=True,
                             callbacks=[checkpoint_callback, lr_monitor],
                             precision=cfg['PRECISION'],
                             )
    trainer.fit(gw_model, dm)

    # lr_finder = trainer.tuner.lr_find(gw_model, dm, num_training=3000)
    # lr_finder.results
    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    # Load Checkpoint from Training
    ckpt_path = get_ckpt_path(wab.wb_logger.name, wab.wb_logger.version, cfg['CKPT'])  # Use this if using wandb
    print(f'{ckpt_path}')
    ckpt_metric = torch.load(ckpt_path['path'])['ckpt_metrics']

    wandb.finish()
    api = wandb.Api()
    run = api.run("/".join(['mdunlap', wab.wb_logger.name, wab.wb_logger.version]))
    run.config['val/loss'] = ckpt_metric['val/loss'].tolist()
    run.config['val/auroc'] = ckpt_metric['val/auroc'].tolist()
    run.config['train/loss'] = ckpt_metric['train/loss'].tolist()
    run.config['train/auroc'] = ckpt_metric['train/auroc'].tolist()
    run.config['epoch'] = ckpt_metric['epoch'].tolist()
    run.update()

    del X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, trainer, gw_model, dm, ckpt_metric, LearningRateMonitor, ModelCheckpoint
    gc.collect()

    """ Inference """
    # Prepare submission sheet:
    sub = pd.read_csv('./Data/g2net-gravitational-wave-detection/sample_submission.csv')
    sub = sub.drop(columns=['target'])
    col_names = ['target_avg_specs']
    col_names = col_names + ['target_' + str(i) for i in range(len(q_transforms))]
    for col_name in col_names:
        sub[col_name] = np.nan

    # Save Submission in Ckpt Directory
    sub_dir = os.path.join(ckpt_path['dir'], cfg['CKPT'])
    os.makedirs(sub_dir, exist_ok=True)

    # TEST_BATCH_SIZE = 128
    for ii in range(len(q_transforms) + 1):
        print(f'TESTING {ii + 1} of {len(q_transforms) + 1}')
        sub_org = pd.read_csv('./Data/g2net-gravitational-wave-detection/sample_submission.csv')
        start_test = time.time()
        if ii == 0:
            q_transform_test_ = q_transforms
        else:
            q_transform_test_ = [q_transforms[ii - 1]]
        col_name = col_names[ii]

        """ Load Inference Model """
        gw_model = GwModel.load_from_checkpoint(ckpt_path['path'])
        gw_model.to('cuda')
        gw_model.eval()
        gw_model.freeze()

        """ Load Testing Data from Disk """
        start = time.time()
        test_data = LoadData(dataset_name=cfg['DATASET'],
                             data_type='test',
                             partial=PARTIAL)
        test_data.load_data()
        end = time.time()
        print(f'Load Test Data Time {end - start}')
        TEST_LEN = len(test_data.data_info['id'])

        test_dataset = GwTestDataset(x=test_data.X,
                                     df_idxs=np.linspace(0, TEST_LEN - 1, TEST_LEN).astype(int).tolist(),
                                     q_transform=q_transform_test_,
                                     img_transform=img_transform,
                                     hpf=hpf,
                                     norms=train_data.stats,
                                     )
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=cfg['BATCH_SIZE'],
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True)

        """ Inference on Test Dataset """
        preds = torch.empty(test_data.X.shape[0]).to('cuda')
        for idx, (test_x, idxs) in enumerate(test_dataloader):
            test_x = test_x.to('cuda')
            pred_logits = gw_model(test_x)
            preds[idxs] = torch.squeeze(pred_logits.sigmoid())
        col_values = sub[col_name].to_numpy()
        col_values[0:len(preds)] = preds.to('cpu').numpy()
        sub[col_name] = col_values
        del gw_model, test_data, test_dataloader, test_dataset, q_transform_test_, col_values
        gc.collect()
        end_test = time.time()
        sub_org.target = sub[col_name]
        sub_org.to_csv(os.path.join(sub_dir, f'submission_{col_name}.csv'), index=False)
        print(f'TEST {ii + 1} of {len(q_transforms) + 1}: {end_test - start_test}')

    # Save Submission in Ckpt Directory
    sub.to_csv(os.path.join(sub_dir, 'submission_all.csv'), index=False)
    sub['target'] = sub[col_names].mean(axis=1)
    sub = sub.drop(columns=col_names, axis=1)
    sub.to_csv(os.path.join(sub_dir, 'submission.csv'), index=False)
    print(f'Saved Submission CSV: {os.path.join(sub_dir, "submission.csv")}')
    return

if __name__ == '__main__':
    cfg_name = sys.argv[1]
    print(f'Starting File: {cfg_name}')
    cfg = load_cfg(cfg_name)
    execute_training(cfg)
    print(f'Completed File: {cfg_name}')

print('End of Script')
