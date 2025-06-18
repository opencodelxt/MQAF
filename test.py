import logging
import os

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import IQADataset
from models.memory import MemoryModel
from options.test_options import TestOptions
from utils.process_image import ToTensor, five_point_crop
from utils.util import setup_seed, set_logging


class Test:
    def __init__(self, config):
        self.opt = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_model()
        self.init_data()
        self.criterion = torch.nn.MSELoss()

    def init_model(self):
        self.model = MemoryModel(num_words=self.opt.memory_size)
        self.model.to(self.device)
        self.load_model()
        self.model.eval()

    def init_data(self):
        test_dataset = IQADataset(
            ref_path=self.opt.val_ref_path,
            dis_path=self.opt.val_dis_path,
            txt_file_name=self.opt.val_list,
            transform=ToTensor(),
        )
        logging.info('number of test scenes: {}'.format(len(test_dataset)))

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=False,
            shuffle=False
        )

    def load_model(self):
        model_path = self.opt.ckpt
        if not os.path.exists(model_path):
            raise ValueError(f'Model not found at {model_path}')

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f'Loaded model from {model_path}')

    def test(self):
        with torch.no_grad():
            losses = []
            pred_epoch = []
            labels_epoch = []
            img_names = []

            with tqdm(desc='Testing', unit='it', total=len(self.test_loader)) as pbar:
                for _, data in enumerate(self.test_loader):
                    pred = 0
                    for i in range(self.opt.num_avg_val):
                        d_img_org = data['d_img_org'].to(self.device)
                        r_img_org = data['r_img_org'].to(self.device)
                        labels = data['score'].view(-1, 1)
                        labels = labels.type(torch.FloatTensor).to(self.device)

                        d_img_org, r_img_org = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)
                        r_img_org = r_img_org if self.opt.train_mode != 'memory_only' else None
                        score = self.model(d_img_org, r_img_org)
                        pred += score

                    pred /= self.opt.num_avg_val
                    loss = self.criterion(pred, labels)
                    losses.append(loss.item())

                    # 保存结果
                    pred_batch_numpy = pred.data.cpu().numpy()
                    labels_batch_numpy = labels.data.cpu().numpy()
                    pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                    labels_epoch = np.append(labels_epoch, labels_batch_numpy)

                    pbar.update()

            # 计算相关系数
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            avg_loss = np.mean(losses)

            # 打印结果
            msg = f'Test Results:\nLoss: {avg_loss:.4f}\nSRCC: {rho_s:.4f}\nPLCC: {rho_p:.4f}'
            print(msg)
            logging.info(msg)

            # 保存预测结果
            results = {
                'predictions': pred_epoch.tolist(),
                'ground_truth': labels_epoch.tolist(),
                'metrics': {
                    'loss': avg_loss,
                    'srcc': rho_s,
                    'plcc': rho_p
                }
            }

            # 保存结果到文件
            save_path = os.path.join(self.opt.checkpoints_dir, f'test_results.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(results, save_path)
            logging.info(f'Results saved to {save_path}')


if __name__ == '__main__':
    config = TestOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    setup_seed(config.seed)
    set_logging(config)
    tester = Test(config)
    tester.test()