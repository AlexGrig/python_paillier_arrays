import pytest
import numpy as np

from phe import paillier
from arrays_phe import ArrayEncryptor, ArrayDecryptor, matr_right_prod, encrypted_array_size_bytes, catchtime, pretty_print_params


@pytest.fixture
def generate_keys():
    
    public_key, private_key = paillier.generate_paillier_keypair() # keys with default size
    
    return (public_key, private_key)




#@pytest.mark.skip(reason="testing skip functionality")
@pytest.mark.parametrize('Precision', [1e1-5,])
@pytest.mark.parametrize('n_jobs', [
    0,
    1,
    10,
    ])
@pytest.mark.parametrize('magnitude', [ 1e5, ])
@pytest.mark.parametrize('n_rows, n_cols', [
    (5, 2),
    (5, 1),
    (1, 5),
    (50, 22),
    (50, 1),
    (1, 50)
])

def test_encryption_and_decryption(generate_keys, Precision, n_jobs, magnitude, n_rows, n_cols):

    public_key, private_key = generate_keys
    
    array_encryptor = ArrayEncryptor(public_key, n_jobs=n_jobs, precision=Precision)
    array_decryptor = ArrayDecryptor(private_key, n_jobs=n_jobs)
    
    
    a1 = np.random.uniform(-magnitude, magnitude, size=(n_rows,n_cols))
    
    a1_encr = array_encryptor.encrypt(a1)
    
    a1_decr = array_decryptor.decrypt(a1_encr)
    
    np.testing.assert_allclose(a1_decr, a1, rtol=0, atol=Precision*10, 
                                       equal_nan=True, err_msg='', verbose=True)
    

    
    
    
    

#@pytest.mark.skip(reason="testing skip functionality")
@pytest.mark.parametrize('Precision', [1e1-5,])
@pytest.mark.parametrize('n_jobs', [
    0,
    1,
    10,
    ])
@pytest.mark.parametrize('magnitude', [ 10, ])
@pytest.mark.parametrize('n_rows, n_cols', [
    (5, 2),
    (5, 1),
    (1, 5),
    (30, 22),
    (30, 1),
    (1, 30)
])
def test_linear_combination(generate_keys, Precision, n_jobs, magnitude, n_rows, n_cols):
    """
    Test linear combination: c1*E[a1] + c2*E[a2]. * - means element-wise multiplication.
    When decrypted this expression should be equal to plain operation: c1*E[a1] + c2*E[a2].
    
    """
    
    public_key, private_key = generate_keys
    
    array_encryptor = ArrayEncryptor(public_key, n_jobs=n_jobs, precision=Precision)
    array_decryptor = ArrayDecryptor(private_key, n_jobs=n_jobs)
    
    
    a1 = np.random.uniform(-magnitude, magnitude, size=(n_rows,n_cols))
    a2 = np.random.uniform(-magnitude, magnitude, size=(n_rows,n_cols))
    c1 = np.random.uniform(-magnitude, magnitude, size=(n_rows,n_cols))
    c2 = np.random.uniform(-magnitude, magnitude, size=(n_rows,n_cols))
    
    a1_encr = array_encryptor.encrypt(a1)
    a2_encr = array_encryptor.encrypt(a2)
    
    prod_1_encr = c1*a1_encr
    prod_2_encr = c2*a2_encr
    
    lin_comb_encr = prod_1_encr + prod_2_encr
    
    lin_comb_decr = array_decryptor.decrypt(lin_comb_encr)
    
    np.testing.assert_allclose(lin_comb_decr, c1*a1 + c2*a2, rtol=0, atol=Precision*10, 
                                       equal_nan=True, err_msg='', verbose=True)


#@pytest.mark.skip(reason="testing skip functionality")
@pytest.mark.parametrize('Precision', [1e1-5,])
@pytest.mark.parametrize('n_jobs', [
    0,
    1,
    10,
    ])
@pytest.mark.parametrize('magnitude', [ 10, ])
@pytest.mark.parametrize('n_rows_1, n_cols_1, n_rows_2, n_cols_2', [
    (5, 2, 2, 3),
    (5, 5, 5, 1),
    #(1, 5),
    (30, 16, 16, 2),
    #(30, 1),
    #(1, 30)
])
def test_matrix_multiplication(generate_keys, Precision, n_jobs, magnitude, n_rows_1, n_cols_1, n_rows_2, n_cols_2):
    
    public_key, private_key = generate_keys
    
    array_encryptor = ArrayEncryptor(public_key, n_jobs=n_jobs, precision=Precision)
    array_decryptor = ArrayDecryptor(private_key, n_jobs=n_jobs)
    
    m1 = np.random.uniform(-magnitude, magnitude, size=(n_rows_1, n_cols_1))
    m2 = np.random.uniform(-magnitude, magnitude, size=(n_rows_2, n_cols_2))
    
    m2_encr = array_encryptor.encrypt(m2)
    
    prod_encr = matr_right_prod(m1,m2_encr, n_jobs)
    prod_decr = array_decryptor.decrypt(prod_encr)
    
    np.testing.assert_allclose(prod_decr, np.dot(m1,m2), rtol=0, atol=Precision*10, 
                                       equal_nan=True, err_msg='', verbose=True)
    
    
    