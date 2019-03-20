# Codes mainly based on Xia Li's code.
# RESCAN: Recurrent Squeeze-and-Excitation Context Aggregation Net
# https://github.com/XiaLiPKU/RESCAN
# Thanks for the support from Xia Li
import os
import sys
import cv2
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image

import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TrainValDataset, TestDataset
from TFN import TFN
from cal_ssim import SSIM, MSSSIM
from utils import transmission, atmospheric_light

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.net = TFN().cuda()
        self.crit = L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.msssim = MSSSIM().cuda()

        self.step = 0
        self.perceptual_weight = settings.perceptual_weight
        self.loss_weight = settings.loss_weight
        self.total_variation_weight = settings.total_variation_weight
        self.ssim_loss_weight = settings.ssim_loss_weight
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}

        self.opt = Adam(self.net.parameters(), lr=settings.lr)
        self.sche = MultiStepLR(self.opt, milestones=[11000, 70000, 90000, 110000, 130000], gamma=0.1)
    
    # using tensorboard for data recording
    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name, train_mode=True):
        dataset = {
            True: TrainValDataset,
            False: TestDataset,
        }[train_mode](dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=True, num_workers=self.num_workers, drop_last=True)
        if train_mode:
            return iter(self.dataloaders[dataset_name])
        else:
            return self.dataloaders[dataset_name]

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict()
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step

    def inf_batch(self, name, batch):
        img_input, img_target, img_original, img_A = batch['img_input'], batch['img_target'], batch['img_original'], batch['img_A']
        img_input, img_target, img_original, img_A = img_input.cuda(), img_target.cuda(), img_original.cuda(), img_A.cuda()
        img_input, img_target, img_original, img_A = Variable(img_input), Variable(img_target), Variable(img_original), Variable(img_A)

        improved_tran = self.net(img_input)

        haze_estimation =  (img_original - img_A) / improved_tran + img_A
        
        # different losses
        mse_loss = MSELoss().cuda()

        loss_list = [self.crit(haze_estimation, img_target)]
        
        ssim_list = [self.ssim(haze_estimation, img_target)]

        msssim_list = [1 - self.msssim(haze_estimation, img_target)]

        total_variation_list = []

        restored = haze_estimation
        reg_loss = torch.sum(torch.abs(restored[:, :, :-1] - restored[:, :, 1:])) + \
               torch.sum(torch.abs(restored[:, :-1, :] - restored[:, 1:, :]))
        reg_loss = reg_loss / (settings.patch_size * settings.patch_size)
        
        total_variation_list.append(reg_loss)

        loss = self.loss_weight * sum(loss_list) \
                + self.total_variation_weight * sum(total_variation_list) \
                + self.ssim_loss_weight * sum(msssim_list)

        all_loss = [loss] 

        losses = {
            'l1-loss ' : loss.data[0]
            for _, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim ' : ssim.data[0]
            for _, ssim in enumerate(ssim_list)
        }
        total_loss = {
            'total_loss ' : loss.data[0]
            for _, loss in enumerate(all_loss)
        }
        losses.update(ssimes)
        losses.update(total_loss)

        return haze_estimation, loss, losses

    def save_image(self, name, num, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        out_img_file = os.path.join(self.log_dir, name + '_output', '%d_%d_%s.png' % (num, self.step, name))
        out_image = pred.numpy()
        out_image = np.transpose(out_image, (1, 2, 0))
        cv2.imwrite(out_img_file, out_image)

        input_img_file = os.path.join(self.log_dir, name + '_input', '%d_%d_%s.png' % (num, self.step, name))
        input_image = data.numpy()
        input_image = np.transpose(input_image, (1, 2, 0))
        cv2.imwrite(input_img_file, input_image)

        target_img_file = os.path.join(self.log_dir, name + '_target', '%d_%d_%s.png' % (num, self.step, name))
        target_image = label.numpy() 
        target_image = np.transpose(target_image, (1, 2, 0))
        cv2.imwrite(target_img_file, target_image)


def run_train_val(ckp_name='latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    sess.tensorboard('train')
    sess.tensorboard('val')

    dt_train = sess.get_dataloader('train')
    dt_val = sess.get_dataloader('val')

    while sess.step < settings.iteration:
        sess.sche.step()

        sess.net.train()
        sess.net.zero_grad()

        batch_t = next(dt_train)
        pred_t, loss_t, losses_t = sess.inf_batch('train', batch_t)
        sess.write('train', losses_t)
        loss_t.backward()
        sess.opt.step()

        if sess.step % 32 == 0:
            sess.net.eval()
            batch_v = next(dt_val)
            pred_v, loss_v, losses_v = sess.inf_batch('val', batch_v)
            sess.write('val', losses_v)

        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints('latest')
        
        if sess.step % int(sess.save_steps / 2) == 0:
            sess.save_image('train', 0, [batch_t['img_original'][0], pred_t[0], batch_t['img_target'][0]])

            logger.info('save image as step_%d' % sess.step)
        
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1

# image test in batch
def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name)

    dt = sess.get_dataloader('test', train_mode=False)

    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        pred, loss, losses = sess.inf_batch('test', batch)
        batch_size = 1
        all_num += batch_size
        sess.save_image('test', i, [batch['img_original'][0], pred[0], batch['img_target'][0]])
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d %s: %f' % (i, key, val))

    for key, val in all_losses.items():
        logger.info('total %s: %f' % (key, val / all_num))

# single image test
def run_real(ckp_name, img_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name)
    
    root_dir = os.path.join(settings.data_dir, settings.real_dir)
    root_output_dir = os.path.join(settings.data_dir, settings.real_output_dir)
    input_file = os.path.join(root_dir, img_name)
    output_file = os.path.join(root_output_dir, img_name[:-4] + '_dehaze.png')

    img = cv2.imread(input_file).astype(np.float32) / 255.0
    
    # step of DCP
    img_trans = transmission(img, omega=0.95, window=1)
    img_input = np.ndarray((1, img_trans.shape[0], img_trans.shape[1]))
    img_input[0] = img_trans
    img_input = img_input.astype(np.float32)
    
    img_A = atmospheric_light(img, window=1)
    temp = np.ndarray((3, img_trans.shape[0], img_trans.shape[1]))
    temp[0] = img_A[0]
    temp[1] = img_A[1]
    temp[2] = img_A[2]
    img_A = temp.astype(np.float32)

    img = np.transpose(img, (2, 0, 1))

    x = torch.rand(1, img_input.shape[0], img_input.shape[1], img_input.shape[2])
    img_input = torch.tensor(img_input)
    x[0] = img_input
    img_input = Variable(x).cuda()
    
    # transmission filtering
    logger.info('Reading Image')
    pred = sess.net(img_input)
    logger.info('Dehazing Image')
    out_image = pred[0].cpu().data.numpy()
    out_image =  (img - img_A) / out_image + img_A
    out_image = out_image * 255.0
    logger.info('Finish Dehazing Image')
    out_image = np.transpose(out_image, (1, 2, 0))

    cv2.imwrite(output_file, out_image)
    logger.info('Saving Image')
    logger.info('Finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='train')
    parser.add_argument('-m', '--model', default='latest')
    parser.add_argument('-n', '--name', default='canyon.png')

    args = parser.parse_args(sys.argv[1:])
    
    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)
    elif args.action == 'real':
        run_real(args.model, args.name)
