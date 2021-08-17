"""
Created on Mon July 11 18:50:39 2020

@author: Haifeng Luo
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import os
import sys
import torch
from torch.optim import Adam, optimizer, lr_scheduler
from tqdm import tqdm
import BatchSampleGenerator as BSG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import py_util
import PDE_net_torch
import Loss_torch


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='IndividualTreeExtraction/pre_trained_PDE_net', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training for each GPU [default: 12]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=50000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate for lr decay [default: 0.95]')
parser.add_argument('--training_data_path',
                    default='datasets/custom_data/PDE/training_data/',
                    #default='/container/directory/data/training_data/',
                    help='Make sure the source training-data files path')
parser.add_argument('--validating_data_path',
                    default='datasets/custom_data/PDE/validating_data/',
                    #default='/container/directory/data/validating_data/',
                    help='Make sure the source validating-data files path')

FLAGS = parser.parse_args()
TRAIN_DATA_PATH = FLAGS.training_data_path
VALIDATION_PATH = FLAGS.validating_data_path


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_data_type = torch.float16 if device=='cuda' else torch.float32

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def train():
    pointclouds = torch.rand( dtype=tensor_data_type, size=(BATCH_SIZE, NUM_POINT, 3), device=device)
    direction_labels = torch.rand( dtype=tensor_data_type, size=(BATCH_SIZE, NUM_POINT, 3), device=device)
    #####DirectionEmbedding
    with torch.cuda.amp.autocast():
        DeepPointwiseDirections = PDE_net_torch.get_model_RRFSegNet('PDE_net',
                                                pointclouds,
                                                is_training=True,
                                                weight_decay=0.0001,
                                                bn_decay=0.0001,
                                                k=20)
    DeepPointwiseDirections.cuda()
    #loss = 1 * loss_esd + 0 * loss_pd
    optomizer = Adam(DeepPointwiseDirections.parameters(), weight_decay=0.0001, lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.ExponentialLR(optomizer, 0.95)
    ###optimizer--Adam

    init_loss = 999.999
    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        ####training data generator
        train_set = py_util.get_data_set(TRAIN_DATA_PATH)
        generator_training = BSG.minibatch_generator(TRAIN_DATA_PATH,BATCH_SIZE, train_set, NUM_POINT)

        ####validating data generator
        val_set = py_util.get_data_set(VALIDATION_PATH)
        generator_val = BSG.minibatch_generator(VALIDATION_PATH, BATCH_SIZE, val_set, NUM_POINT)

        #####trainging steps
        temp_loss = train_one_epoch(DeepPointwiseDirections, epoch, train_set, generator_training, optomizer, scaler, scheduler)
        torch.cuda.empty_cache()
        #####validating steps
        validation(DeepPointwiseDirections, val_set, generator_val)

        if (temp_loss < init_loss) or (epoch%5==0):
            torch.save({
            'epoch': epoch,
            'model_state_dict': DeepPointwiseDirections.state_dict(),
            'optimizer_state_dict': optomizer.state_dict(),
            'loss': temp_loss,
            }, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.pt'))
            init_loss = temp_loss


def train_one_epoch(model, epoch, train_set, generator, opt, scaler, scheduler):
    """ ops: dict mapping from string to tf ops """

    num_batches_training = len(train_set) // (BATCH_SIZE)
    print('-----------------training--------------------')
    print('training steps: %d'%num_batches_training)

    total_loss = 0
    total_loss_esd = 0
    total_loss_pd = 0
    for i in tqdm(range(num_batches_training)):
        ###
        opt.zero_grad()
        batch_train_data, batch_direction_label_data, _ = next(generator)
        model.train()
        with torch.cuda.amp.autocast():
            batch_train_data = torch.tensor(batch_train_data, device=device)
            batch_direction_label_data = torch.tensor(batch_direction_label_data, device=device)
            y = model(batch_train_data)
            loss_esd_ = Loss_torch.slack_based_direction_loss(y,batch_direction_label_data)
            loss_pd_ = Loss_torch.direction_loss(y,batch_direction_label_data)
            total_loss = 1 * loss_esd_ + 0 * loss_pd_
        
        scaler.scale(total_loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if i % 20 == 0:
            print('loss: %f, loss_esd: %f,loss_pd: %f'%(total_loss, loss_esd_, loss_pd_))

    print('trianing_log_epoch_%d'%epoch)
    log_string('epoch: %d, loss: %f, loss_esd: %f,loss_pd: %f'%(epoch, total_loss/(num_batches_training),
                                                                total_loss_esd/(num_batches_training),
                                                                total_loss_pd/(num_batches_training)))
    return total_loss.cpu()/(num_batches_training)


def validation(model, test_set, generator):

    num_batches_testing = len(test_set) // (BATCH_SIZE)
    total_loss = 0
    total_loss_esd = 0
    total_loss_pd = 0
    for _ in tqdm(range(num_batches_testing)):
        ###
        batch_test_data, batch_direction_label_data, _ = next(generator)
        ###
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                y = model(torch.as_tensor(batch_test_data).cuda())
                batch_direction_label_data = torch.as_tensor(batch_direction_label_data).cuda()
                loss_esd_ = Loss_torch.slack_based_direction_loss(y,batch_direction_label_data)
                loss_pd_ = Loss_torch.direction_loss(y,batch_direction_label_data)
        total_loss_esd = loss_esd_

    log_string('val loss: %f, loss_esd: %f, loss_pd: %f'%(total_loss_esd/num_batches_testing,
                                                          loss_esd_/num_batches_testing,
                                                          loss_pd_/num_batches_testing))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
