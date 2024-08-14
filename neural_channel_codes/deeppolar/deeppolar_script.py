import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import csv
import json
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

from ..channels import Channel
from ..utils import snr_db2sigma, errors_ber, errors_bler

def deeppolar_test(polar, device, deeppolar=None, config=None):
    """Test example for DeepPolar.
    
    Parameters
    ----------
    polar : PolarCode
        Polar code object.
    device : torch.device
        Device to use for computations.
        Eg: torch.device('cuda:0') or torch.device('cpu')
    net : CRISP_RNN, optional
        CRISP-RNN model object.
        If None, then default model is used.
    config : dict, optional
        Configuration dictionary.
        Example config provided as `deepcommpy/crisp/test_config.json`.
        """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config is None:
        with open(os.path.join(script_dir, 'test_config.json'), 'r') as f:
            config = json.load(f)

    snr_range = config['snr_range']
    num_test_batches = config['test_size'] // config['test_batch_size']
    noise_type = config['noise_type']
    assert noise_type in ['awgn', 'fading', 'radar', 't-dist'], "Please choose one of these noise types: 'awgn', 'fading', 'radar', 't-dist'"

    channel = Channel(noise_type)
    bers_KO_test = [0. for _ in snr_range]
    blers_KO_test = [0. for _ in snr_range]

    bers_SC_test = [0. for _ in snr_range]
    blers_SC_test = [0. for _ in snr_range]

    kernel = config['N'] == deeppolar.ell

    # with torch.no_grad():
    #     for k in range(num_test_batches):
    #         msg_bits = 2*torch.randint(0, 2, (config['test_batch_size'], polar.K), dtype=torch.float) - 1
    #         msg_bits = msg_bits.to(device)
    #         polar_code = polar.encode(msg_bits)
    #         for snr_ind, snr in enumerate(snr_range):
    #             sigma = snr_db2sigma(snr)
    #             noisy_code = channel.corrupt_signal(polar_code, sigma, vv = config['vv'], radar_power = config['radar_power'], radar_prob = config['radar_prob'])
    #             noise = noisy_code - polar_code

    #             SC_llrs, decoded_SC_msg_bits = polar.sc_decode(noisy_code, snr)
    #             ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign())
    #             bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign())

    #             decoded_RNN_msg_bits = polar.crisp_rnn_decode(noisy_code, net)

    #             ber_RNN = errors_ber(msg_bits, decoded_RNN_msg_bits.sign())
    #             bler_RNN = errors_bler(msg_bits, decoded_RNN_msg_bits.sign())


    #             bers_RNN_test[snr_ind] += ber_RNN/num_test_batches
    #             bers_SC_test[snr_ind] += ber_SC/num_test_batches


    #             blers_RNN_test[snr_ind] += bler_RNN/num_test_batches
    #             blers_SC_test[snr_ind] += bler_SC/num_test_batches

    with torch.no_grad():
        try:
            while min(total_block_errors_SC, total_block_errors_KO) <= num_errors:
                msg_bits = 2 * (torch.rand(args.test_batch_size, args.K) < 0.5).float() - 1
                msg_bits = msg_bits.to(device)
                polar_code = polar.encode_plotkin(msg_bits)

                if 'KO' in args.encoder_type:
                    if kernel:
                        KO_polar_code = KO.kernel_encode(args.kernel_size, KO.gnet_dict[1][0], msg_bits, info_positions, binary=binary)
                    else:
                        KO_polar_code = KO.deeppolar_encode(msg_bits, binary=binary)

                noisy_code = polar.channel(polar_code, snr, noise_type)
                noise = noisy_code - polar_code
                noisy_KO_code = KO_polar_code + noise if 'KO' in args.encoder_type else noisy_code

                SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, snr)
                ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign()).item()
                bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign()).item()
                total_block_errors_SC += int(bler_SC*args.test_batch_size)
                if 'KO' in args.decoder_type:
                    if kernel:
                        if args.decoder_type == 'KO_parallel':
                            KO_llrs, decoded_KO_msg_bits = KO.kernel_parallel_decode(args.kernel_size, KO.fnet_dict[1][0], noisy_KO_code, info_positions)
                        else:
                            KO_llrs, decoded_KO_msg_bits = KO.kernel_decode(args.kernel_size, KO.fnet_dict[1][0], noisy_KO_code, info_positions)
                    else:
                        KO_llrs, decoded_KO_msg_bits = KO.deeppolar_decode(noisy_KO_code)
                else:  # if SC is also used for KO
                    KO_llrs, decoded_KO_msg_bits = KO.sc_decode_new(noisy_KO_code, snr)

                ber_KO = errors_ber(msg_bits, decoded_KO_msg_bits.sign()).item()
                bler_KO = errors_bler(msg_bits, decoded_KO_msg_bits.sign()).item()
                total_block_errors_KO += int(bler_KO*args.test_batch_size)

                batches_processed += 1

                # Update accumulative results for logging
                bers_KO_test[snr_ind] += ber_KO
                bers_SC_test[snr_ind] += ber_SC
                blers_KO_test[snr_ind] += bler_KO
                blers_SC_test[snr_ind] += bler_SC

                # Real-time logging for progress, updating in-place
                print(f"SNR: {snr} dB, Sigma: {sigma:.5f}, SC_BER: {bers_SC_test[snr_ind]/batches_processed:.6f}, SC_BLER: {blers_SC_test[snr_ind]/batches_processed:.6f}, KO_BER: {bers_KO_test[snr_ind]/batches_processed:.6f}, KO_BLER: {blers_KO_test[snr_ind]/batches_processed:.6f}, Batches: {batches_processed}", end='\r')

        except KeyboardInterrupt:
            # print("\nInterrupted by user. Finalizing current SNR...")
            pass

        # Normalize cumulative metrics by the number of processed batches for accuracy
        bers_KO_test[snr_ind] /= (batches_processed + 0.00000001)
        bers_SC_test[snr_ind] /= (batches_processed + 0.00000001)
        blers_KO_test[snr_ind] /= (batches_processed + 0.00000001)
        blers_SC_test[snr_ind] /= (batches_processed + 0.00000001)
        print(f"SNR: {snr} dB, Sigma: {sigma:.5f}, SC_BER: {bers_SC_test[snr_ind]:.6f}, SC_BLER: {blers_SC_test[snr_ind]:.6f}, KO_BER: {bers_KO_test[snr_ind]:.6f}, KO_BLER: {blers_KO_test[snr_ind]:.6f}")

    return bers_RNN_test, blers_RNN_test, bers_SC_test, blers_SC_test
