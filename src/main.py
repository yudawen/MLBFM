from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from lib.datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

from torch.utils.data.dataloader import default_collate
import warnings
warnings.filterwarnings('ignore')


def dance_collator(batch):

    ret = {'input': default_collate([b['input'] for b in batch])}
    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    # detection
    hm = default_collate([b['hm'] for b in batch])


    reg_mask = default_collate([b['reg_mask'] for b in batch])
    ind = default_collate([b['ind'] for b in batch])
    wh = default_collate([b['wh'] for b in batch])
    reg = default_collate([b['reg'] for b in batch])


    ret .update( {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,'reg': reg})#','contour':contour

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    # print(ct_num)

    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.bool)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1

    point_num=64

    # polygon evolution
    init_box = torch.zeros([batch_size, ct_num, point_num, 2], dtype=torch.float)
    targ_poly1 = torch.zeros([batch_size, ct_num, point_num, 2],dtype=torch.float)
    target_corner = torch.zeros([batch_size, ct_num, point_num],dtype=torch.float)
    targ_poly2 = torch.zeros([batch_size, ct_num, point_num, 2],dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num],dtype=torch.float)


    whs = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)

    keymasks = torch.zeros([batch_size, ct_num, point_num],dtype=torch.float)

    if ct_num != 0:
        init_box[ct_01] = torch.Tensor(sum([b['init_box'] for b in batch], []))
        targ_poly1[ct_01] = torch.Tensor(sum([b['targ_poly1'] for b in batch], []))
        whs[ct_01] = torch.Tensor(sum([b['whs'] for b in batch], []))
        target_corner[ct_01] = torch.Tensor(sum([b['target_corner'] for b in batch], []))
        ct_cls[ct_01] = torch.Tensor(sum([b['ct_cls'] for b in batch], []))
        keymasks[ct_01]=torch.Tensor(sum([b['keymasks'] for b in batch], []))

    evolution = {'init_box': init_box,'targ_poly1': targ_poly1, 'whs': whs,'ct_01': ct_01.float(),'keymasks':keymasks,'target_corner':target_corner}
    ret.update(evolution)
    ret.update({'ct_cls':ct_cls})


    return ret

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  print('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=opt.batch_size,
      shuffle=False,
      num_workers=0,
      pin_memory=True,
      collate_fn=dance_collator
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True,
     collate_fn = dance_collator

  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                   epoch, model)
    elif epoch%5==0:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      # save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
      #            epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    if epoch==opt.num_epochs:
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model, optimizer)
  logger.close()

if __name__ == '__main__':
    import time

    opt = opts().parse()
    main(opt)