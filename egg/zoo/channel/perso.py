# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.channel.features import OneHotLoader, UniformLoader
from egg.zoo.channel.archs import Sender, Receiver
from egg.core.util import dump_sender_receiver_test
from egg.core.util import dump_impose_message
from egg.core.reinforce_wrappers import RnnReceiverImpatient
from egg.core.reinforce_wrappers import SenderImpatientReceiverRnnReinforce
from egg.core.util import dump_sender_receiver_impatient


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--dim_dataset', type=int, default=10240,
                        help='Dim of constructing the data (default: 10240)')
    parser.add_argument('--force_eos', type=int, default=0,
                        help='Force EOS at the end of the messages (default: 0)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--receiver_num_heads', type=int, default=8,
                        help='Number of attention heads for Transformer Receiver (default: 8)')
    parser.add_argument('--sender_num_heads', type=int, default=8,
                        help='Number of self-attention heads for Transformer Sender (default: 8)')
    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--causal_sender', default=False, action='store_true')
    parser.add_argument('--causal_receiver', default=False, action='store_true')

    parser.add_argument('--sender_generate_style', type=str, default='in-place', choices=['standard', 'in-place'],
                        help='How the next symbol is generated within the TransformerDecoder (default: in-place)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: rnn)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument('--probs', type=str, default='uniform',
                        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="Penalty for the message length, each symbol would before <EOS> would be "
                             "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")
    parser.add_argument('--early_stopping_thr', type=float, default=0.9999,
                        help="Early stopping threshold on accuracy (default: 0.9999)")

    parser.add_argument('--receiver_weights',type=str ,default="receiver_weights.pth",
                        help="Weights of the receiver agent")
    parser.add_argument('--sender_weights',type=str ,default="sender_weights.pth",
                        help="Weights of the sender agent")
    parser.add_argument('--save_dir',type=str ,default="analysis/",
                        help="Directory to save the results of the analysis")
    parser.add_argument('--impatient', type=bool, default=False,
                        help="Impatient listener")
    parser.add_argument('--unigram_pen', type=float, default=0.0,
                        help="Add a penalty for redundancy")

    parser.add_argument('--p_train', type=float, default=0.0,
                        help='Probability of corruption used to train the network')


    args = core.init(parser, params)

    return args



def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}


def get_rob_accuracy(game, n_features, device, gs_mode,save_dir,n_repeat=20):

    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()
    powerlaw_probs=torch.tensor(powerlaw_probs,device='cuda')
    
    all_P=np.linspace(0,1,51)

    all_unif_acc=np.zeros(len(all_P))
    all_powerlaw_acc=np.zeros(len(all_P))

    for _ in range(n_repeat):

        for k,p_test in enumerate(all_P):

            game.sender.set_noise(p_test)
            game.sender.test_with_noise=True

            sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
                dump_sender_receiver_impatient(game, dataset, gs=gs_mode, device=device, variable_length=True, test_mode=True,save_dir=save_dir,verbose=False)

            logits=torch.stack(receiver_outputs)
            acc_vec=torch.exp(torch.diagonal(logits))

            all_unif_acc[k]+=torch.mean(acc_vec).cpu().numpy().item()
            all_powerlaw_acc[k]+=torch.sum(acc_vec*powerlaw_probs).cpu().numpy().item()


    return all_powerlaw_acc/n_repeat,all_unif_acc/n_repeat


def main(params):
    opts = get_params(params)
    force_eos = opts.force_eos == 1
    device = opts.device

    ### ------------------  LOAD MODEL  ------------------ ###

    receiver_weights=f"all_dir_save/dir_save_{opts.p_train}/receiver/receiver_weights500.pth"
    sender_weights=f"all_dir_save/dir_save_{opts.p_train}/sender/sender_weights500.pth"
    
    sender = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
    sender = core.RnnSenderReinforce(sender,
                                opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                force_eos=force_eos)
    

    receiver = Receiver(n_features=opts.receiver_hidden, n_hidden=opts.vocab_size)
    receiver = RnnReceiverImpatient(receiver, opts.vocab_size, opts.receiver_embedding,opts.receiver_hidden, cell=opts.receiver_cell, 
                                    num_layers=opts.receiver_num_layers, max_len=opts.max_len, n_features=opts.n_features)
    
    sender.load_state_dict(torch.load(sender_weights,map_location=torch.device('cpu')))
    receiver.load_state_dict(torch.load(receiver_weights,map_location=torch.device('cpu')))


    game = SenderImpatientReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                                receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                                length_cost=opts.length_cost,unigram_penalty=opts.unigram_pen).to(device=device)

    ### ------------------  COMPUTE ROBUST ACCURACY  ------------------ ###

    rob_acc_powerlaw,rob_acc_uni=get_rob_accuracy(game, opts.n_features, device, False,save_dir=opts.save_dir)

    np.save(opts.save_dir+"robust_uni_acc_"+str(opts.p_train)+".npy",rob_acc_uni)
    np.save(opts.save_dir+"robust_powerlaw_acc_"+str(opts.p_train)+".npy",rob_acc_powerlaw)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
