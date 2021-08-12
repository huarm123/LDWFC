#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torch
import argparse
import torch.nn.functional as F
from Imagefolder_modified import Imagefolder_modified
from resnet import ResNet18_Normalized, ResNet50_Normalized
from bcnn import BCNN_Normalized
import self_correcter
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time

torch.manual_seed(0)
torch.cuda.manual_seed(0)

class Manager_AM(object):
    def __init__(self, options):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter
            path    [dict]  path of the dataset and model
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = options
        self._path = options['path']
        os.popen('mkdir -p ' + self._path)
        self._data_base = options['data_base']
        self._smooth = options['smooth'] #True / False
        self._droprate = options['droprate']
        self._labelnoise = options['labelnoise']
        self._tk = options['tk']
        self._label_weight = options['label_weight']
        self._denoise = options['denoise']
        self._class = options['n_classes']
        self._step = options['step']
        self._resume = False
        self._queue_size = options['queue_size']
        self._warm_up = options['warm_up']
        print('class number: {}\t\t denoise: {}\t\t resume: {}\t\t drop rate: {}\t\t smooth label: {}\t\t label weight: {}\t\t tk: {}'.format(self._class, self._denoise, self._resume, self._droprate, self._smooth, self._label_weight,self._tk))
        # Network
        if options['net'] == 'resnet18':
            NET = ResNet18_Normalized
        elif options['net'] == 'resnet50':
            NET = ResNet50_Normalized
        elif options['net'] == 'bcnn':
            NET = BCNN_Normalized
        else:
            raise AssertionError('Not implemented yet')

        if self._step == 1:
            net = NET(n_classes=options['n_classes'], pretrained=True)
        elif self._step == 2:
            net = NET(n_classes=options['n_classes'], pretrained=False)
        else:
            raise AssertionError('Wrong step')
        if torch.cuda.device_count() >= 1:
            self._net = torch.nn.DataParallel(net).cuda()
            print('cuda device : ', torch.cuda.device_count())
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Optimizer
        if options['net'] == 'bcnn':
            if self._step == 1:
                params_to_optimize = self._net.module.fc.parameters()
                print('step1')
            else:
                params_to_optimize = self._net.parameters()
        else:
            params_to_optimize = self._net.parameters()
        self._optimizer = torch.optim.SGD(params_to_optimize, lr=self._options['base_lr'],
                                          momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # Load data
        data_dir = self._data_base
        self._train_data = Imagefolder_modified(os.path.join(data_dir, 'train'), transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
        print('number of classes in trainset is : {}'.format(len(self._train_data.classes)))
        print('number of classes in testset is : {}'.format(len(test_data.classes)))
        assert len(self._train_data.classes) == options['n_classes'] and len(test_data.classes) == options['n_classes'], 'number of classes is wrong'
        self._train_loader = DataLoader(self._train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=16,
                                       shuffle=False, num_workers=4, pin_memory=True)
        # print('len(self._train_data)',len(self._train_data))

        self.correcter = self_correcter.Correcter(len(self._train_data), self._class, self._queue_size)


    def _save_checkpoint(self, state, filename='checkpoint.pth'):
        torch.save(state, filename)

    def _selection(self, false_id, ids, weighted_cos_angle, labels,t):
        id_batch = ids.numpy().tolist()
        loss_update = [id_batch.index(x) for x in id_batch if x not in false_id]
        logits_final = weighted_cos_angle[loss_update]
        labels_final = labels[loss_update]

        hard_id = []

        value = self.correcter.separate_clean_and_unclean_keys(self._labelnoise, false_id)
        prec = torch.LongTensor(self.correcter.prec.tolist()).cuda()

        for i in range(len(false_id)):
            id = false_id[i]
            f_x = self.correcter.certainty_array[id]
            if f_x < value:    #低不确定性
                hard_id.append(id)

        #if t > self._warm_up:
            #loss_update1 = [id_batch.index(x) for x in id_batch if x  in hard_id]
            #logits_final = torch.cat((logits_final, weighted_cos_angle[loss_update1]), 0)
            #labels_final = torch.cat((labels_final, prec[loss_update1]), 0)

        if self._smooth == True:
            loss = self._smooth_label_loss(logits_final, labels_final)
        else:
            loss = self._criterion(logits_final, labels_final)
        return loss, len(logits_final)


    def _smooth_label_loss(self,logits,labels):
        N = labels.size(0)
        smoothed_labels = torch.full(size=(N, self._class),
                                     fill_value=(1 - self._label_weight) / (self._class - 1)).cuda()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=self._label_weight)
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -torch.sum(log_prob * smoothed_labels) / N
        return loss

    def train(self):
        """
        Train the network
        """
        # step = args.step
        print('===> About training in a two-step process! ===')
        print('------\n'
              'drop rate: [{}]\t'
              '\n------'.format(self._droprate))

        print('Training ... ')

        loadmodel = 'checkpoint1.pth'
        logfile = 'training_log1.txt'

        # check if it is resume mode
        print('-----------------------------------------------------------------------------')
        if self._resume == True:
            print('resume')
            assert os.path.isfile(loadmodel), 'please make sure checkpoint.pth exists'
            print('---> loading checkpoint.pth <---')
            checkpoint = torch.load(loadmodel)
            assert self._step == checkpoint['step'], 'step in checkpoint does not match step in argument'
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            best_epoch = checkpoint['best_epoch']
            self._net.load_state_dict(checkpoint['bnn_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self.correcter.all_predictions = (checkpoint['all_predictions'])
            # self.correcter.softmax_record = (checkpoint['softmax_record'])
            self.correcter.update_counters = (checkpoint['update_counters'])

        else:
            if self._step == 2:
                print('--->        step2 checkpoint loaded         <---')
                self._net.load_state_dict(torch.load('model/bcnn.pth'))
            else:
                print('--->        no checkpoint loaded         <---')

            start_epoch = 0
            best_accuracy = 0.0
            best_epoch = None

        print('-----------------------------------------------------------------------------')

        with open(logfile, "a") as f:
            f.write('------ Step: {} ...\n'.format(self._step))
            f.write('------\n'
                    'drop rate: [{}]\tqueue_size: [{}]\t'
                    'warm_up: [{}]\tinit_lr: [{}]\t'
                    'labelnoise: [{}]\t '
                    '\n'.format(self._droprate, self._queue_size, self._warm_up, self._options['base_lr'], self._labelnoise))

        # print('Epoch\tTrain Loss\tTrain Accuracy\tTest Accuracy\tEpoch Runtime')
        s = 30
        false_id=[]
        for t in range(start_epoch,self._options['epochs']):
            epoch_start = time.time()

            epoch_loss = []
            record=[]
            num_correct = 0
            num_total = 0
            num_train_total = 0
            # self._classweigt_tmp = torch.zeros(self._class).cuda()
            for it, (X, y, id, path) in enumerate(self._train_loader):

                # Enable training mode
                self._net.train(True)
                # Data
                X = X.cuda()
                y = y.cuda()
                # Clear the existing gradients
                self._optimizer.zero_grad()
                # Forward pass
                cos_angle = self._net(X)  # score is in shape (N, 200)
                # pytorch only takes label as [0, num_classes) to calculate loss
                cos_angle = torch.clamp(cos_angle, min=-1, max=1)
                weighted_cos_angle = s * cos_angle
                _, prec = torch.max(cos_angle.data, 1)
                softmax_matrix = prec.cpu().detach().numpy()  # 用来算准确度
                self.correcter.async_update_prediction_matrix(id, softmax_matrix)

                if self._denoise == True:
                    if t < 2:
                        if self._smooth == False:
                            loss = self._criterion(weighted_cos_angle, y)
                        else:
                            #smooth label loss
                            loss = self._smooth_label_loss(weighted_cos_angle, y)
                        num_train = y.size(0)
                    else:
                        #loss after sample selection
                        self.correcter.get_certainty_array(id)
                        loss, num_train= self._selection(false_id, id, weighted_cos_angle, y,t)
                else:
                    if self._smooth == False:
                        loss = self._criterion(weighted_cos_angle, y)
                    else:
                        # smooth label loss
                        loss = self._smooth_label_loss(weighted_cos_angle, y)
                    num_train = y.size(0)

                epoch_loss.append(loss.item())
                # Prediction
                closest_dis, prediction = torch.max(cos_angle.data, 1)

                #record cos_angle and image id
                for i in range(y.size(0)):
                    temp=[]
                    temp.append(cos_angle[i,y[i]].clone().detach())
                    temp.append(id[i].clone())
                    record.append(temp)

                # prediction is the index location of the maximum value found,
                num_total += y.size(0)  # y.size(0) is the batch size
                train_correct_batch = torch.sum(prediction == y.data).item()
                num_correct += torch.sum(prediction == y.data).item()
                num_train_total += num_train
                # Backward
                loss.backward()
                self._optimizer.step()
                epoch_end = time.time()
                print('Epoch:[{0:03d}/{1:03d}]  Iter:[{2:04d}/{3:04d}]'
                      'Train Accuracy :[{4:6.2f}]'
                      'Iter Runtime:[{5:6.2f}]'
                      'training number:[{6:03d}]'.format(
                    t + 1, self._options['epochs'], it + 1, len(self._train_data) // self._options['batch_size'],
                    (1.0 * train_correct_batch) / y.size(0),
                    epoch_end - epoch_start, num_total, ))
            # Record the train accuracy of each epoch
            train_accuracy = 100 * num_correct / num_total
            test_accuracy = self.test(self._test_loader)
            self._scheduler.step(test_accuracy)  # the scheduler adjust lr based on test_accuracy

            record.sort(key=lambda x: x[0])#ascending order
            all_id = [int(x[1]) for x in record]

            num_drop = int(min(t / self._tk, 1) * self._droprate * len(all_id))
            # num_drop = int(self._droprate * len(all_id))
            false_id = all_id[:num_drop]

            epoch_end = time.time()

            torch.save(self._net.state_dict(), 'model/resnet_epoch{}.pth'.format((t + 1)))

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = t + 1  # t starts from 0
                print('*', end='')
                # Save mode
                torch.save(self._net.state_dict(), os.path.join(self._path, options['net'] + '.pth'))
            print("all_predictions", len(self.correcter.all_predictions[0]))

            print("update_counters", self.correcter.update_counters[0])
            self._save_checkpoint({
                'epoch': t + 1,
                'bnn_state_dict': self._net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'best_epoch': best_epoch,
                'best_accuracy': best_accuracy,
                'step': self._step,
                'all_predictions': self.correcter.all_predictions,
                # 'softmax_record': self.correcter.softmax_record,
                'update_counters': self.correcter.update_counters
            }, filename=loadmodel)

            # print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f\t\t%4.2f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
            #                                                 train_accuracy, test_accuracy,
            #                                                 epoch_end - epoch_start, num_train_total))

            print('------\n'
                  'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t'
                  'Test Accuracy: [{:6.2f}]\t'
                  'Epoch Runtime: [{:6.2f}]\t' \
                  '\n------'.format(
                t + 1, self._options['epochs'], train_accuracy, test_accuracy, epoch_end - epoch_start))
            with open(logfile, "a") as f:
                output = 'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t' \
                         'Test Accuracy: [{:6.2f}]\t' \
                         'Epoch Runtime: [{:7.2f}]\tTrain_total[{:06d}]\tclean_key[{:06d}]'.format(
                    t + 1, self._options['epochs'], train_accuracy, test_accuracy, (epoch_end - epoch_start),
                    num_train_total, len(self.correcter.clean_key))
                f.write(output + "\n")

        print('-----------------------------------------------------------------')
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net.train(False) # set the mode to evaluation phase
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()
                # Prediction
                score = self._net(X)
                _, prediction = torch.max(score, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # set the mode to training phase
        return 100 * num_correct / num_total


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--net', dest='net', type=str, default='resnet18',
                        help='supported options: resnet18, resnet50, bcnn')
    parser.add_argument('--n_classes', dest='n_classes', type=int, default=200,
                        help='number of classes')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    parser.add_argument('--w_decay', dest='w_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=80)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--path', dest='path', type=str, default='model')
    parser.add_argument('--droprate', dest='droprate', type=float, default=0.25,
                        help='droprate')
    parser.add_argument('--labelnoise', dest='labelnoise', type=float, default=0.5,
                        help='labelnoise')
    parser.add_argument('--denoise', dest='denoise', type=bool, default=False)
    parser.add_argument('--resume', dest='resume', type=bool, default=False)
    parser.add_argument('--smooth', dest='smooth', type=bool, default=False)
    parser.add_argument('--label_weight', dest='label_weight', type=float, default=0.5)
    parser.add_argument('--data_base', dest='data_base', type=str)
    parser.add_argument('--tk', dest='tk', type=int, default=5)
    parser.add_argument('--queue_size', dest='queue_size', type=int, default=10)
    parser.add_argument('--warm_up', dest='warm_up', type=int, default=15)
    parser.add_argument('--step', dest='step', type=int, default=1,
                        help='Step 1 is training fc only; step 2 is training the entire network')

    args = parser.parse_args()

    model = args.path

    print(os.path.join(os.popen('pwd').read().strip(), model))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), model)):
        print('>>>>>> Creating directory \'model\' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), model))

    path = os.path.join(os.popen('pwd').read().strip(), model)

    options = {
        'base_lr': args.lr,
        'weight_decay': args.w_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'path': path,
        'data_base': args.data_base,
        'net': args.net,
        'n_classes': args.n_classes,
        'droprate': args.droprate,
        'labelnoise': args.labelnoise,
        'denoise': args.denoise,
        'resume': args.resume,
        'smooth': args.smooth,
        'label_weight': args.label_weight,
        'tk': args.tk,
        'step': args.step,
        'queue_size': args.queue_size,
        'warm_up': args.warm_up
    }
    manager = Manager_AM(options)
    manager.train()