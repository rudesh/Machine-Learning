import binascii
from typing import Tuple

from Crypto import Random
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.PublicKey import RSA
from Crypto.Util import Counter


# noinspection PyMethodMayBeStatic
class EncryptionHandler:
    def __init__(self, key_filename: str = "config/mykey.pem"):
        self.BLOCK_SIZE = 16
        """
        If .pem file exists, read keys from it.
        If it doesn't exist, generate 2048 bit RSA key-pair, save it to a .pem file
        """
        try:
            f = open(key_filename)
            self.__private_key = RSA.importKey(f.read())
            self.public_key = self.__private_key.publickey()
            f.close()
        except FileNotFoundError:
            self.__private_key = RSA.generate(2048, Random.new().read)
            self.public_key = self.__private_key.publickey()
            f = open(key_filename, 'wb')
            f.write(self.__private_key.exportKey('PEM'))
            f.close()

    def encrypt_message(self, msg, recipient_public_key) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt message using AES with CTR mode and encrypt the AES key with recipient's public key and return iv,
        encrypted message and encrypted AES key as a tuple.

        Using module Crypto.Cipher.PKCS1_OAEP as cryptographic padding is needed, and RSA.encrypt(self, plaintext, K))
        is not recommended as it may lead to security vulnerabilities. Module Crypto.Cipher.AES used for message encryption.

        Solution taken from https://stackoverflow.com/a/44662262
        """

        # Generate the initialization vector and symmteric key for AES
        iv, key = Random.new().read(self.BLOCK_SIZE), Random.new().read(self.BLOCK_SIZE)
        # Convert the IV to a Python integer
        iv_int = int(binascii.hexlify(iv), 16)

        # Create AES-CTR cipher.
        aes = AES.new(key, AES.MODE_CTR, counter=Counter.new(AES.block_size * 8, initial_value=iv_int))

        # Encrypt and return IV, ciphertext and the encrypted key
        return iv, aes.encrypt(msg), PKCS1_OAEP.new(recipient_public_key).encrypt(key)

    def decrypt_message(self, iv: bytes, ciphertext: bytes, encrypted_key: bytes) -> str:
        """
        Decrypt an incoming ciphertext.
        """

        # Decrypt the encrypted AES key
        decryption = PKCS1_OAEP.new(self.__private_key)
        key = decryption.decrypt(encrypted_key)

        # Initialize counter for decryption.
        iv_int = int(binascii.hexlify(iv), 16)
        aes = AES.new(key, AES.MODE_CTR, counter=Counter.new(self.BLOCK_SIZE * 8, initial_value=iv_int))

        # Return the decrypted message
        return aes.decrypt(ciphertext).decode("utf8")

    def get_public_key(self):
        return self.public_key
