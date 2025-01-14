# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import os
import time
import toml
import torch
import random
import numpy as np
import math
from dataset import AudioToTextDataLayer
from helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch,  \
                    add_ctc_labels, AmpOptimizations, model_multi_gpu, print_dict, \
                    print_once, make_bn_layers_in_eval_mode
from quartznet_model import AudioPreprocessing, CTCLossNM, GreedyCTCDecoder, Jasper
from optimizers import Novograd, AdamW
from torch.utils.tensorboard import SummaryWriter
import copy

import pdb

import warnings
warnings.filterwarnings("ignore")

def lr_policy(initial_lr, step, N):
    """
    learning rate decay
    Args:
        initial_lr: base learning rate
        step: current iteration number
        N: total number of iterations over which learning rate is decayed
    """
    min_lr = 1e-10
    res = initial_lr * ((N - step) / N) ** 2
    return max(res, min_lr)

def warmup_decay_policy(initial_lr, step, N, warmup_portion=0.1):
    min_lr = 1e-10
    step +=1 # 1 indexed
    warmup_steps = math.floor(warmup_portion*N)
    remaining_steps = N - warmup_steps

    if step <= warmup_steps:
        return initial_lr * (step/warmup_steps)
    else:
        res = initial_lr * ((N-step)/remaining_steps)
        return max(res,min_lr)


def save(model, optimizer, epoch, output_dir, save_optimizer=True):
    """
    Saves model checkpoint
    Args:
        model: model
        optimizer: optimizer
        epoch: epoch of model training
        output_dir: path to save model checkpoint
    """
    os.makedirs(output_dir,exist_ok=True)
    class_name = model.__class__.__name__
    unix_time = time.time()
    #file_name = "{0}_{1}-epoch-{2}.pt".format(class_name, unix_time, epoch)
    file_name = "{0}.pt".format(class_name)
    print_once("Saving module {0} in {1}".format(class_name, os.path.join(output_dir, file_name)))
    if (not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        save_checkpoint={
                        'epoch': epoch,
                        'state_dict': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict() if save_optimizer else None
                        }

        torch.save(save_checkpoint, os.path.join(output_dir, file_name))
    print_once('Saved.')




def train(
        data_layer,
        data_layer_eval,
        model,
        ctc_loss,
        greedy_decoder,
        optimizer,
        optim_level,
        labels,
        multi_gpu,
        args,
        fn_lr_policy=None,
        other_inputs=None,
        device=torch.device("cpu")):
    """Trains model
    Args:
        data_layer: training data layer
        data_layer_eval: evaluation data layer
        model: model ( encapsulates data processing, encoder, decoder)
        ctc_loss: loss function
        greedy_decoder: greedy ctc decoder
        optimizer: optimizer
        optim_level: AMP optimization level
        labels: list of output labels
        multi_gpu: true if multi gpu training
        args: script input argument list
        fn_lr_policy: learning rate adjustment function
    """
    def eval():
        """Evaluates model on evaluation dataset
        """
        with torch.no_grad():
            _global_var_dict = {
                'EvalLoss': [],
                'predictions': [],
                'transcripts': [],
            }
            eval_dataloader = data_layer_eval.data_iterator
            for data in eval_dataloader:
                tensors = []
                for d in data:
                    if isinstance(d, torch.Tensor):
                        tensors.append(d.to(device))
                    else:
                        tensors.append(d)
                t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = tensors

                model.eval()
                
                if optim_level == 1:
                  with amp.disable_casts():
                      pass
                    #   t_processed_signal_e, t_processed_sig_length_e = audio_preprocessor(t_audio_signal_e, t_a_sig_length_e) 
                else:
                  t_processed_signal_e, t_processed_sig_length_e = audio_preprocessor(t_audio_signal_e, t_a_sig_length_e)
                
                if jasper_encoder.use_conv_mask:
                    t_log_probs_e, t_encoded_len_e = model.forward((t_processed_signal_e, t_processed_sig_length_e))
                else:
                    t_log_probs_e = model.forward(t_processed_signal_e)

                t_loss_e = ctc_loss(log_probs=t_log_probs_e, targets=t_transcript_e, input_length=t_encoded_len_e, target_length=t_transcript_len_e)

                t_predictions_e = greedy_decoder(log_probs=t_log_probs_e)

                values_dict = dict(
                    loss=[t_loss_e],
                    predictions=[t_predictions_e],
                    transcript=[t_transcript_e],
                    transcript_length=[t_transcript_len_e]
                )
                process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

            # final aggregation across all workers and minibatches) and logging of results
            wer, cer, eloss = process_evaluation_epoch(_global_var_dict)

            if args.use_cer:
                print_once("==========>>>>>>Evaluation Loss: {0}".format(eloss))
                print_once("==========>>>>>>Evaluation CER: {0}\n".format(cer))
                return cer, eloss
            else:
                print_once("==========>>>>>>Evaluation Loss: {0}".format(eloss))
                print_once("==========>>>>>>Evaluation WER: {0}\n".format(wer))
                return wer, eloss

    print_once("Starting .....")
    start_time = time.time()
    train_dataloader = data_layer.data_iterator
    epoch = args.start_epoch
    step = epoch * args.step_per_epoch
    patience = 0

    audio_preprocessor = model.module.audio_preprocessor if hasattr(model, 'module') else model.audio_preprocessor
    data_spectr_augmentation = model.module.data_spectr_augmentation if hasattr(model, 'module') else model.data_spectr_augmentation
    jasper_encoder = model.module.jasper_encoder if hasattr(model, 'module') else model.jasper_encoder

    print_once("Pre Evaluation ....................... ......  ... .. . .")
    prev_best_wer=10000
    #prev_best_wer,_ = eval()
    prev_best_epoch_wer = prev_best_wer

    while True:
        if multi_gpu:
            data_layer.sampler.set_epoch(epoch)
        print_once("Starting epoch {0}, step {1}".format(epoch, step))
        last_epoch_start = time.time()
        batch_counter = 0
        average_loss = 0
        for data in train_dataloader:
            #torch.cuda.empty_cache()
            tensors = []
            for d in data:
                if isinstance(d, torch.Tensor):
                    tensors.append(d.to(device))
                else:
                    tensors.append(d)

            if batch_counter == 0:

                if fn_lr_policy is not None:
                    adjusted_lr = fn_lr_policy(step)
                    for param_group in optimizer.param_groups:
                            param_group['lr'] = adjusted_lr
                optimizer.zero_grad()
                last_iter_start = time.time()
            t_audio_signal_t, t_a_sig_length_t, t_transcript_t, t_transcript_len_t = tensors     
            model.train()
            if args.turn_bn_eval:
                make_bn_layers_in_eval_mode(model.jasper_encoder)
            if optim_level == 1:
                pass
            #   with amp.disable_casts():
            #       t_processed_signal_t, t_processed_sig_length_t = audio_preprocessor(t_audio_signal_t, t_a_sig_length_t) 
            else:
              t_processed_signal_t, t_processed_sig_length_t = audio_preprocessor(t_audio_signal_t, t_a_sig_length_t)

            t_processed_signal_t = data_spectr_augmentation(t_processed_signal_t)
            if jasper_encoder.use_conv_mask:
                t_log_probs_t, t_encoded_len_t = model.forward((t_processed_signal_t, t_processed_sig_length_t))
            else:
                t_log_probs_t = model.forward(t_processed_signal_t)

            t_loss_t = ctc_loss(log_probs=t_log_probs_t, 
                                targets=t_transcript_t, 
                                input_length=t_encoded_len_t, 
                                target_length=t_transcript_len_t)
            t_total_loss = t_loss_t

            if args.gradient_accumulation_steps > 1:
                t_total_loss = t_total_loss / args.gradient_accumulation_steps

            if optim_level >=0 and optim_level <=3:
                pass
                # with amp.scale_loss(t_total_loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
            else:
                t_total_loss.backward()
            batch_counter += 1
            average_loss += t_total_loss.item()

            #pdb.set_trace()
            if batch_counter % args.gradient_accumulation_steps == 0:
                optimizer.step()
                #print()

                if step % args.train_frequency == 0:
                    t_predictions_t = greedy_decoder(log_probs=t_log_probs_t)
                    e_tensors = [t_predictions_t, t_transcript_t, t_transcript_len_t]
                    train_wer, train_wer_list = monitor_asr_train_progress(e_tensors, labels=labels)
                    print_once("Loss@Step: {0}  ::::::: {1}".format(step, str(average_loss)))
                    print_once("Step time: {0} seconds".format(time.time() - last_iter_start))
                    other_inputs["summary_writer"].add_scalar('Loss/train', average_loss, step)

                if step > 0 and step % args.eval_frequency == 0:
                    if args.save_after_each_epoch == False:
                        print_once("Doing Evaluation ....................... ......  ... .. . .")
                        e_wer,e_loss = eval()
                        other_inputs["summary_writer"].add_scalar('Loss/eval',e_loss,step)
                        other_inputs["summary_writer"].add_scalar('WER/eval',e_wer,step)
                        if prev_best_wer is None or e_wer < prev_best_wer:
                            
                            prev_best_wer = e_wer
                            save(model, optimizer,epoch,output_dir=args.best_dir, save_optimizer=False)
                        else:
                            pass
                step += 1
                batch_counter = 0
                average_loss = 0
                if args.num_steps is not None and step >= args.num_steps:
                    break

        if args.num_steps is not None and step >= args.num_steps:
            break
        print_once("Finished epoch {0} in {1}".format(epoch, time.time() - last_epoch_start))
        epoch += 1
        save(model, optimizer,epoch,output_dir=args.output_dir)

        if args.save_after_each_epoch:
            print_once("Doing Evaluation after finishing epoch ....................... ......  ... .. . .")
            e_wer,e_loss = eval()
            other_inputs["summary_writer"].add_scalar('Loss/eval',e_loss,step)
            other_inputs["summary_writer"].add_scalar('WER/eval',e_wer,step)

        if prev_best_epoch_wer is None or e_wer < prev_best_epoch_wer+ 0.01:
            patience = 0
            if e_wer < prev_best_epoch_wer:
                prev_best_epoch_wer = e_wer
                if args.save_after_each_epoch:
                    save(model, optimizer,epoch,output_dir=args.best_dir, save_optimizer=False)
        else:
            #pass            
            patience +=1
            if patience >= args.early_stop_patience:
                assert patience == args.early_stop_patience
                print_once("Patience: "+str(patience))
                print_once("Stopping Early BBYE!! ......  ... .. . .")
                break
        print_once("Patience: "+str(patience))
            
        if args.num_steps is None and epoch >= args.num_epochs:
            break

    print_once("Done in {0}".format(time.time() - start_time))
    print_once("Final Evaluation ....................... ......  ... .. . .")
    eval()

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #assert(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = args.cudnn
    # set up distributed training
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        print_once("DISTRIBUTED TRAINING with {} gpus".format(torch.distributed.get_world_size()))             
    # define amp optimiation level
    if args.fp16:
        optim_level = 1
    else:
        optim_level = 0

    jasper_model_definition = toml.load(args.model_toml)
    dataset_vocab = jasper_model_definition['labels']['labels']
    ctc_vocab = add_ctc_labels(dataset_vocab)

    train_manifest = args.train_manifest
    val_manifest = args.val_manifest
    featurizer_config = jasper_model_definition['input']
    featurizer_config_eval = jasper_model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level
    featurizer_config_eval["optimization_level"] = optim_level

    sampler_type = featurizer_config.get("sampler", 'default')
    perturb_config = jasper_model_definition.get('perturb', None)
    if args.pad_to_max:
        assert(args.max_duration > 0)
        featurizer_config['max_duration'] = args.max_duration
        featurizer_config_eval['max_duration'] = args.max_duration
        featurizer_config['pad_to'] = -1        
        featurizer_config_eval['pad_to'] = -1
        
    print_once('model_config')
    print_dict(jasper_model_definition)

    if args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid gradient accumulation steps parameter {}'.format(args.gradient_accumulation_steps))
    if args.batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError('gradient accumulation step {} is not divisible by batch size {}'.format(args.gradient_accumulation_steps, args.batch_size))

    if args.min_duration is not None:
        featurizer_config['min_duration'] = args.min_duration
        featurizer_config_eval['min_duration'] = args.min_duration
        

    data_layer = AudioToTextDataLayer(
                                    dataset_dir=args.wav_dir,
                                    featurizer_config=featurizer_config,
                                    perturb_config=perturb_config,
                                    manifest_filepath=train_manifest,
                                    labels=dataset_vocab,
                                    batch_size=args.batch_size // args.gradient_accumulation_steps,
                                    multi_gpu=multi_gpu,
                                    pad_to_max=args.pad_to_max,
                                    sampler=sampler_type
                                    )

    data_layer_eval = AudioToTextDataLayer(
                                    dataset_dir=args.wav_dir,
                                    featurizer_config=featurizer_config_eval,
                                    manifest_filepath=val_manifest,
                                    labels=dataset_vocab,
                                    batch_size=args.batch_size,
                                    multi_gpu=multi_gpu,
                                    pad_to_max=args.pad_to_max,
                                    shuffle=False
                                    )

    jasper_model_definition['turn_bn_eval']=args.turn_bn_eval
    model = Jasper(feature_config=featurizer_config,
                 jasper_model_definition=jasper_model_definition,
                 feat_in=1024,
                 num_classes=len(ctc_vocab))

    audio_preprocessor = model.module.audio_preprocessor if hasattr(model, 'module') else model.audio_preprocessor

    other_inputs = {} #other inputs to be sent to train() e.g. intial decoder parameters

    if args.ckpt is not None:
        print_once("loading model from {}".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        for k in audio_preprocessor.state_dict().keys():
            #print(k)
            #if k not in checkpoint['state_dict']:
                #print("\taudio_preprocessor." + k)
            checkpoint['state_dict'][k] = checkpoint['state_dict'].pop("audio_preprocessor." + k)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if args.resume_from_ckpt:
            args.start_epoch = checkpoint['epoch']
        else:
            args.start_epoch = 0
    else:
        args.start_epoch = 0

    if args.freeze_first_n_blocks > 0:
        for i,block in enumerate(model.jasper_encoder.encoder):
            if i<args.freeze_first_n_blocks:
                for param in block.parameters():
                    param.requires_grad=False

    ctc_loss = CTCLossNM(num_classes=len(ctc_vocab), 
                        zero_infinity=args.zero_infinity, 
                        )
    
    greedy_decoder = GreedyCTCDecoder()

    print_once("Number of parameters in encoder: {0}".format(model.jasper_encoder.num_weights()))
    print_once("Number of parameters in decode: {0}".format(model.jasper_decoder.num_weights()))

    N = len(data_layer)
    if sampler_type == 'default':
        args.step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
    elif sampler_type == 'bucket':
        args.step_per_epoch = int(len(data_layer.sampler) / args.batch_size )

    print_once('-----------------')
    print_once('Have {0} examples to train on.'.format(N))
    print_once('Have {0} steps / (gpu * epoch).'.format(args.step_per_epoch))
    print_once('-----------------')

    if args.lr_decay == 'decay':
        fn_lr_policy = lambda s: lr_policy(args.lr, s, args.num_epochs * args.step_per_epoch)
    elif args.lr_decay == 'warmup':
        fn_lr_policy = lambda s: warmup_decay_policy(args.lr, s, args.num_epochs * args.step_per_epoch)
    else:
        fn_lr_policy = None


    model.to(device)
    if 'pretrained_model' in other_inputs:
        other_inputs['pretrained_model'].to(device)

    trainable_parameters = model.parameters()
    if args.optimizer_kind == "novograd":
        optimizer = Novograd(trainable_parameters,
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.optimizer_kind == "adam":
        optimizer = AdamW(trainable_parameters,
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.optimizer_kind == "sgd":
        optimizer = torch.optim.SGD(trainable_parameters,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise ValueError("invalid optimizer choice: {}".format(args.optimizer_kind))


    if optim_level >= 0 and optim_level <=3:
        pass
        # model, optimizer = amp.initialize(
        #     min_loss_scale=1.0,
        #     models=model,
        #     optimizers=optimizer,
        #     opt_level=AmpOptimizations[optim_level])
    model = model_multi_gpu(model, multi_gpu)

    if args.ckpt is not None and args.load_optimizer_state:
        optimizer.load_state_dict(checkpoint['optimizer'])

    

    os.makedirs(os.path.join(args.output_dir,"runs"),exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir,"runs"))
    other_inputs["summary_writer"]=summary_writer

    train(data_layer, data_layer_eval, model, \
          ctc_loss=ctc_loss, \
          greedy_decoder=greedy_decoder, \
          optimizer=optimizer, \
          labels=ctc_vocab, \
          optim_level=optim_level, \
          multi_gpu=multi_gpu, \
          fn_lr_policy=fn_lr_policy, \
          args=args, \
          other_inputs=other_inputs,
          device=device)

def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help='data batch size')
    parser.add_argument("--num_epochs", default=10, type=int, help='number of training epochs. if number of steps if specified will overwrite this')
    parser.add_argument("--num_steps", default=None, type=int, help='if specified overwrites num_epochs and will only train for this number of iterations')
    parser.add_argument("--save_freq", dest="save_frequency", default=300, type=int, help='number of epochs until saving checkpoint. will save at the end of training too.')
    parser.add_argument("--eval_freq", dest="eval_frequency", default=200, type=int, help='number of iterations until doing evaluation on full dataset')
    parser.add_argument("--train_freq", dest="train_frequency", default=25, type=int, help='number of iterations until printing training statistics on the past iteration')
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help='weight decay rate')
    parser.add_argument("--train_manifest", type=str, required=True, help='relative path given dataset folder of training manifest file')
    parser.add_argument("--wav_dir", type=str, required=True, help='path to directory containing wav files in manifest')
    parser.add_argument("--model_toml", type=str, required=True, help='relative path given dataset folder of model configuration file')
    parser.add_argument("--val_manifest", type=str, required=True, help='relative path given dataset folder of evaluation manifest file')
    parser.add_argument("--max_duration", type=float, help='maximum duration of audio samples for training and evaluation')
    parser.add_argument("--min_duration", type=float, default=None, help='minimum duration of audio samples for training and evaluation')
    parser.add_argument("--pad_to_max", action="store_true", default=False, help="pad sequence to max_duration")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help='number of accumulation steps')
    parser.add_argument("--optimizer", dest="optimizer_kind", default="novograd", type=str, help='optimizer')
    parser.add_argument("--lr_decay", type=str, default='none', choices=['warmup','decay','none'], help='learning rate decay strategy')
    parser.add_argument("--cudnn", action="store_true", default=False, help="enable cudnn benchmark")
    parser.add_argument("--fp16", action="store_true", default=False, help="use mixed precision training")
    parser.add_argument("--output_dir", type=str, required=True, help='saves results in this directory')
    parser.add_argument("--best_dir", type=str, required=True, help='saves the best ckpt in this directory')
    parser.add_argument("--ckpt", default=None, type=str, help="path to load a pre-trained ckpt")
    parser.add_argument("--resume_from_ckpt",action="store_true",default=False, help="resume training from ckpt's epoch")
    parser.add_argument("--load_optimizer_state",action="store_true",default=False, help="load optimizer's state from ckpt")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--early_stop_patience", default=3, type=int, help='max number of epochs to wait if performance does not improve')
    parser.add_argument("--turn_bn_eval",action='store_true',default=False,help='fix mean and variance of bn layers during training')
    parser.add_argument("--zero_infinity",action='store_true',default=False,help='turn on zero infinity in CTC loss')
    parser.add_argument("--save_after_each_epoch", action='store_true', default=False,help='save best ckpt only after each epoch')
    parser.add_argument("--freeze_first_n_blocks",type=int,default=0,help='freeze first n blocks in the jasper encoder')
    parser.add_argument("--use_cer", action='store_true', default=False, help='use cer for evaluation')
    args=parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    print_dict(vars(args))
    main(args)
