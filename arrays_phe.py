import sys
import math
import fractions
from sys import getsizeof
import numpy as np
from functools import partial
from time import perf_counter
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from joblib import Parallel, delayed

from phe import encoding
from phe import paillier

Precision = 1e-15

def is_encrypted(arr):
    one_array_element = np.take(arr, 0)
    if isinstance(one_array_element, paillier.EncryptedNumber):
        return True
    else:
        return False
    
class ArrayEncryptor():
    
    def __init__(self, public_key, n_jobs=2, precision=1e-12):
        assert public_key is not None, "public_key can not be None"
        assert precision is not None, "precision can not be None"
        
        self.public_key = public_key
        self.precision = precision
        self.n_jobs = n_jobs
        
        self.enc_partial_function = partial(self.public_key.encrypt, precision=precision)
        self.enc_encoded_partial_function = lambda xx: self.public_key.encrypt_encoded(xx,1)
        # numpy.vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
        self.vec_encrypt = np.vectorize(self.enc_partial_function, otypes=None, doc=None, excluded=None, 
                                        cache=False, signature=None)
        
        self.vec_encrypt_encoded = np.vectorize(self.enc_encoded_partial_function, otypes=None, doc=None, excluded=None, 
                                        cache=False, signature=None)
        
    def encrypt(self, arr, which='float'):
        
        """
        which (string): which method of Paillier public key to use. If `float` use encrypt method, 
                        if `encoded` use encrypt_encoded
        """
        
        if isinstance(arr, paillier.EncryptedNumber): # the same function can encrypt single numbers as well
            use_func = self.enc_partial_function if (which=='float') else self.enc_encoded_partial_function
            return use_func(arr)
        elif isinstance(arr, np.ndarray):
            use_func = self.vec_encrypt if (which=='float') else self.vec_encrypt_encoded
            
            n_jobs_eff = min( (arr.size // 3) + 1, self.n_jobs )
            
            if (arr.size < 10) or (self.n_jobs == 0) or (self.n_jobs == 1) or \
                (n_jobs_eff == 0) or (n_jobs_eff == 1):
                return use_func(arr)
            else:
                #import pdb; pdb.set_trace()
                orig_shape = arr.shape
                
                with Parallel(n_jobs_eff) as p:
                    res = np.concatenate(
                                p(delayed(use_func)(x) for x in np.array_split(arr.reshape(-1), n_jobs_eff))
                              )
                res = res.reshape(orig_shape)
                return res
                #res = Parallel(n_jobs=6)(delayed(self.enc_partial_function)(i ** 2) for i in range(10))
        else:
            raise ValueError(f"Unsupported type {type(arr)}")
            
class ArrayDecryptor():
    
    def __init__(self, secret_key, n_jobs=2):
        assert secret_key is not None, "public_key can not be None"
        
        self.secret_key = secret_key
        self.n_jobs = n_jobs
        
        # numpy.vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
        self.vec_decrypt = np.vectorize(self.secret_key.decrypt, otypes=None, doc=None, excluded=None, 
                                        cache=False, signature=None)
        
        self.vec_decrypt_encoded = np.vectorize(self.secret_key.decrypt_encoded, otypes=None, doc=None, excluded=None, 
                                        cache=False, signature=None)
        
    def decrypt(self, arr, which='float'):
        """
        which (string): which method of Paillier public key to use. If `float` use encrypt method, 
                        if `encoded` use encrypt_encoded
        """
        
        use_func = self.vec_decrypt if (which=='float') else self.vec_decrypt_encoded
        #import pdb; pdb.set_trace()
        
        if isinstance(arr, paillier.EncryptedNumber): # the same function can encrypt single numbers as well
            return use_func(arr)
        elif isinstance(arr, np.ndarray):
            
            n_jobs_eff = min( (arr.size // 3) + 1, self.n_jobs )
            
            if (arr.size < 10) or (self.n_jobs == 0) or (self.n_jobs == 1) or \
                (n_jobs_eff == 0) or (n_jobs_eff == 1):
                return use_func(arr)
            else:
                
                orig_shape = arr.shape
                
                with Parallel(n_jobs_eff) as p:
                    res = np.concatenate(
                                p(delayed(use_func)(x) for x in np.array_split(arr.reshape(-1), n_jobs_eff))
                              )
                res = res.reshape(orig_shape)
                return res
        else:
            raise ValueError(f"Unsupported type {type(arr)}")

def matr_right_prod(plain_arr, cipher_arr, n_jobs=0):
    
    assert plain_arr.shape[-1] == cipher_arr.shape[0], f"Arrays' shapes must be suitable for matrix product {plain_arr.shape}, {cipher_arr}"
    
    if n_jobs == 0:
        res = np.dot(plain_arr, cipher_arr)
    else:
        n_jobs_eff = min( (plain_arr.shape[0] // 3) + 1, n_jobs )
        if n_jobs_eff == 0:
            return np.dot(plain_arr, cipher_arr)
        
        #import pdb; pdb.set_trace()
        #print(f'n_jobs_eff: {n_jobs_eff}')
        with Parallel(n_jobs) as p:
            res = np.concatenate(
                        p(delayed(np.dot)(x, cipher_arr) for x in np.array_split(plain_arr, n_jobs_eff))
                      )
    return res

def add_encrypted(arr_1, arr_2, n_jobs=0):
    """
    Either both or one of arrays are encrypted.
    """
    
    if n_jobs == 0:
        res = arr_1 + arr_2
    else:
        n_jobs_eff = min( (arr_1.shape[0] // 3) + 1, n_jobs )
        if n_jobs_eff == 0:
            return arr_1 + arr_2
        
        assert arr_1.shape == arr_2.shape, f"Arrays' shapes must be equal for encrypted addition {arr_1.shape}, {arr_2.shape}"
        orig_shape = arr_1.shape
        
        add = lambda x1, x2 : x1 + x2
        with Parallel(n_jobs) as p:
            res = np.concatenate(
                      p(delayed(add)(x, y) for x,y in zip(np.array_split(arr_1.reshape(-1), n_jobs_eff),
                                                                np.array_split(arr_2.reshape(-1), n_jobs_eff)) )
                      )
        res = res.reshape(orig_shape)        
    return res

def mult_encrypted_and_scalar(arr_1, scalar, n_jobs=0):
    """
    Either both or one of arrays are encrypted.
    """
    
    if n_jobs == 0:
        res = arr_1*scalar
    else:
        n_jobs_eff = min( (arr_1.shape[0] // 3) + 1, n_jobs )
        if n_jobs_eff == 0:
            return arr_1*scalar
        
        orig_shape = arr_1.shape
        prod = lambda x, scalar: scalar * x
        
        with Parallel(n_jobs) as p:
            res = np.concatenate(
                      p(delayed(prod)(x, scalar) for x in np.array_split(arr_1.reshape(-1), n_jobs_eff))
                      )
        #import pdb; pdb.set_trace()
        res = res.reshape(orig_shape)        
    return res


def encrypted_array_size_bytes(arr):
    
    #import pdb; pdb.set_trace()
    array_number_elements = arr.size
    
    one_array_element = np.take(arr, 0)
    
    size_of_ciphertext = getsizeof(one_array_element.ciphertext())
    size_of_exponent = getsizeof(one_array_element.exponent)
    
    size_of_array_overhead = getsizeof(arr)
    
    size_of_data = array_number_elements * (size_of_ciphertext + size_of_exponent)
    total_size_in_bytes = size_of_data + size_of_array_overhead
    
    return total_size_in_bytes, size_of_data

def increase_exponent(required_exponent, xx: encoding.EncodedNumber):
    #import pdb; pdb.set_trace()
    
    if xx.exponent < required_exponent:
        base_power = required_exponent-xx.exponent
    else:
        return xx
    
    new_encoding = xx.encoding // pow(xx.BASE,base_power)
    
    new_exponent = required_exponent
        
    if (xx.encoding <= xx.public_key.max_int): # Positive
        
        return encoding.EncodedNumber(xx.public_key, new_encoding % xx.public_key.n, new_exponent)
    elif (xx.encoding >= xx.public_key.n - xx.public_key.max_int):
        # Negative
        mantissa = xx.public_key.n - xx.encoding
        new_mantissa = mantissa // pow(xx.BASE,base_power)
        mantissa = xx.public_key.n - new_mantissa
        
        return encoding.EncodedNumber(xx.public_key, mantissa % xx.public_key.n, new_exponent)
    else:
        raise OverflowError('Overflow detected in decoded number')

def increase_exponent_array(arr, required_exponent):
    """
    
    
    """
    
    
    #import pdb; pdb.set_trace()
    orig_shape = arr.shape
    
    increase_exponent_partial = partial(increase_exponent, required_exponent)
        # numpy.vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
    vect_increase_exponent = np.vectorize(increase_exponent_partial, otypes=None, doc=None, excluded=None, 
                                        cache=False, signature=None)
    
    res = vect_increase_exponent(arr)
    res = res.reshape(orig_shape)
    
    return res

class catchtime:
    def __init__(self, print_out=False, info=''):
        self.print_out = print_out
        self.info = info
        
    def __enter__(self, print_out=False):
        if self.print_out:
            print(f'Enter: {self.info}')
            
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'{self.info} Time: {self.time:.3f} seconds'
        if self.print_out:
            print(self.readout)

def pretty_print_params(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + f'{key}:', flush=True)
            pretty_print_params(value, indent+1)
        else:
            print('\t' * indent + f'{key}: {value}', flush=True)