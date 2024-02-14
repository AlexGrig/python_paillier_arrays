import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

from phe import paillier
from arrays_phe import ArrayEncryptor, ArrayDecryptor, encrypted_array_size_bytes, catchtime, pretty_print_params


Precision = 1e-15
test_default_array_sizes = [100,500,1000]

def speed_and_space_test():
    arg_parser = argparse.ArgumentParser(allow_abbrev=False)
    arg_parser.add_argument('--n_jobs', type=int, required=False)
    arg_parser.add_argument('--arr_sizes', type=int, nargs='+', required=False)
    
    args = {k:v for k,v in vars(arg_parser.parse_args()).items() if v is not None}
    
    array_sizes = test_default_array_sizes if (args.get('arr_sizes', None) is None) else args['arr_sizes']
    n_jobs = 10 if (args.get('n_jobs', None) is None) else args['n_jobs']
    
    print(f'Testing array sizes: {array_sizes}')
    print(f'Testing n_jobs: {n_jobs}')
    print()
    
    public_key, private_key = paillier.generate_paillier_keypair()
    array_encryptor = ArrayEncryptor(public_key, n_jobs=n_jobs, precision=Precision)
    array_decryptor = ArrayDecryptor(private_key, n_jobs=n_jobs)
    
    statistics = defaultdict(list)
    for array_size in array_sizes:
        print(f'Testing array size: {array_size}')
        statistics['Array size'].append(array_size)
        a1 = np.random.uniform(-1e+5, 1e+5, size=(array_size,1))
        a2 = np.random.uniform(-1e+5, 1e+5, size=(array_size,1))

        with catchtime() as time_encryption:
            a1_enc = array_encryptor.encrypt(a1)

        statistics['Encryption time (sec.)'].append( time_encryption.time )
        statistics['Array space (Mb)'].append(a1.nbytes / 1e6)
        statistics['Encrypted space (Mb)'].append(encrypted_array_size_bytes(a1_enc)[0] / 1e6)


        a2_enc = array_encryptor.encrypt(a2)


        with catchtime() as time_addition:
            a1_plus_a2_enc = a1_enc + a2_enc
        statistics['Homomorphic Addition time (sec.)'].append( time_addition.time )

        with catchtime() as time_decryption:
            a1_plus_a2_decr = array_decryptor.decrypt(a1_plus_a2_enc)
        statistics['Decryption time (sec.)'].append( time_decryption.time )

        try:
            np.testing.assert_allclose(a1_plus_a2_decr, a1+a2, rtol=0, atol=Precision*10, 
                                       equal_nan=True, err_msg='', verbose=True)
            statistics['Results are close'].append(True)
        except AssertionError as ae:
            statistics['Results are close'].append(False)
    
    print('')
    print('Test results:')
    pretty_print_params(statistics, indent=1)
    
    
    # Plot test results ->
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    _ = ax1.plot(statistics['Array size'], statistics['Encryption time (sec.)'], 'o-', label='Encryption time (sec.)')
    _ = ax1.plot(statistics['Array size'], statistics['Homomorphic Addition time (sec.)'], 'o-', label='Homomorphic Addition time (sec.)')
    _ = ax1.plot(statistics['Array size'], statistics['Decryption time (sec.)'], 'o-', label='Decryption time (sec.)')
    #ax2 = ax1.twinx)(y
    #_ = ax2.plot(xx, train_statistics['test_rmse'], 'red')

    #ax1.plot((xx[0], xx[-1]), (separate_res['full_model'], separate_res['full_model']), label='Full model')

    #ax1.set_ylim([0.92, 0.97])
    # Plot beauties 
    labels_fontdict = {'fontsize': 15, 'fontweight' : 'demibold'}
    _ = plt.setp(ax1.get_xticklabels(), fontsize=14, fontweight="bold")
    _ = plt.setp(ax1.get_yticklabels(), fontsize=14, fontweight="bold")
    ax1.set_xlabel('Array size', **labels_fontdict)
    ax1.set_ylabel('Time (Sec.)', **labels_fontdict)
    ax1.legend(loc='upper left',fontsize=11)
    plt.savefig("time_statistics.png")
    
    
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    _ = ax1.plot(statistics['Array size'], statistics['Array space (Mb)'], 'o-', label='Array space (Mb)')
    _ = ax1.plot(statistics['Array size'], statistics['Encrypted space (Mb)'], 'o-', label='Encrypted space (Mb)')
    #ax1.set_yscale('log')
    #ax2 = ax1.twinx()
    #_ = ax2.plot(xx, train_statistics['test_rmse'], 'red')

    #_ = ax2.plot(statistics['Array size'], statistics['Encrypted space (Mb)'], 'go-', label='Encrypted space (Mb)')

    #ax1.set_ylim([0.92, 0.97])
    # Plot beauties 
    labels_fontdict = {'fontsize': 15, 'fontweight' : 'demibold'}
    _ = plt.setp(ax1.get_xticklabels(), fontsize=14, fontweight="bold")
    _ = plt.setp(ax1.get_yticklabels(), fontsize=14, fontweight="bold")
    ax1.set_xlabel('Array size', **labels_fontdict)
    ax1.set_ylabel('Size (Mb)', **labels_fontdict)
    ax1.legend(loc='upper left',fontsize=11)
    plt.savefig("space_statistics.png")
    # Plot test results <-
    
if __name__ == '__main__':
    speed_and_space_test()