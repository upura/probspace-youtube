from efficientnet_pytorch import EfficientNet
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader

from src.datasets import SimpleDataset
from src.models import Squeeze
from src.utils import seed_everything
from src.runner import CustomRunner, RMSELoss


if __name__ == '__main__':

    run_name = 'nn000'
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    INPUT_DIR = '../input/'
    train = pd.read_csv(INPUT_DIR + 'train_data.csv', nrows=128)
    test = pd.read_csv(INPUT_DIR + 'test_data.csv', nrows=128)

    TRAIN_IMG_PATH = '../input/img_tr/'
    TEST_IMG_PATH = '../input/img_te/'

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    cv_scores = []

    train_paths = [TRAIN_IMG_PATH + f'{x}.jpg' for x in train['id'].values]
    train_labels = np.log1p(train['y'])

    test_paths = [TEST_IMG_PATH + f'{x}.jpg' for x in test['id'].values]
    test_dataset = SimpleDataset(test_paths, labels=None, transform=None)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=512)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(train, train['ratings_disabled'])):

        X_tr, y_tr = [d for i, d in enumerate(train_paths) if i in list(tr_idx)], train_labels.iloc[tr_idx].values
        X_val, y_val = [d for i, d in enumerate(train_paths) if i in list(va_idx)], train_labels.iloc[va_idx].values

        train_dataset = SimpleDataset(X_tr, y_tr, transform=None)
        valid_dataset = SimpleDataset(X_val, y_tr, transform=None)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
        valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=512)

        loaders = {'train': train_loader, 'valid': valid_loader}
        runner = CustomRunner(device=device)

        model = EfficientNet.from_pretrained('efficientnet-b3')
        model._fc = torch.nn.Sequential(
            torch.nn.Linear(1536, 1),
            Squeeze(),
        )

        criterion = RMSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        logdir = f'../output/logdir_{run_name}/fold{fold_id}'
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=30,
            verbose=True,
        )

        pred = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                       runner.predict_loader(
                                           loader=valid_loader,
                                           resume=f'{logdir}/checkpoints/best.pth',
                                           model=model,),)))

        oof_preds[va_idx] = pred
        score = np.sqrt(mean_squared_error(y_val, pred))
        cv_scores.append(score)
        print('score', score)

        pred = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                       runner.predict_loader(
                                       loader=test_loader,
                                       resume=f'{logdir}/checkpoints/best.pth',
                                       model=model,),)))
        test_preds += pred / 5

    # save results
    print(cv_scores)
    OUTPUT_DIR = '../output/'
    pd.DataFrame(oof_preds).to_csv(OUTPUT_DIR + 'pred/oof.csv', index=False)
    sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
    sub['y'] = np.expm1(test_preds)
    sub.to_csv(OUTPUT_DIR + f'submissions/submission_{run_name}.csv', index=False)
