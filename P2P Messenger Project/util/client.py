from encryption.encryptionmodule import EncryptionHandler
from packet import ClientID, Address
from datetime import datetime, timedelta
import uuid
import yaml
from packet import Port
import logging

logger = logging.getLogger(__name__)


class Client:
    enc: EncryptionHandler

    def __init__(self, uuid: ClientID, name: str, address: Address, pubkey, last_active: datetime, enc=None):
        self.uuid = uuid
        self.name = name
        self.address = address
        self.pubkey = pubkey
        self.enc = enc
        self.last_active = last_active
        self.discoverer = None

    def set_last_active_time(self, last_active=None):
        if not last_active:
            last_active = datetime.now()

        self.last_active = last_active

    def __str__(self):
        return f"Client(uuid={self.uuid}, name={self.name}, address={self.address}, last_active={self.last_active})"

    @property
    def display_str(self) -> str:
        now = datetime.now()
        delta = now - self.last_active
        delta -= delta % timedelta(seconds=1)
        discoverer_tag = f" [{self.discoverer.tag}]" if self.discoverer else ""
        return f"{self.name} ({delta}){discoverer_tag}"

    @staticmethod
    def create_user():  # -> (Client, EncryptionHandler):
        print("Hello and welcome to epykS!")
        name = input("Please enter a username: ")
        uuid_ = uuid.uuid4()
        data = {"uuid": uuid_,
                "name": name,
                "udp_ip": "0.0.0.0",
                "udp_port": 0,
                "key_file": "config/mykey.pem"}

        enc = EncryptionHandler("config/mykey.pem")

        with open('config/user_data.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        return Client(uuid_, name, ("0.0.0.0", Port(0)), enc.public_key, None), enc


    @staticmethod
    def get_user_data(user_data_filename):  # -> (Client, EncryptionHandler):
        """
        Loads user data from a .yml file and creates a User and an EncryptionHandler object.
        :param user_data_filename: the path/name of the .yml file
        :return: a tuple of a User object and an EncryptionHandler object
        """
        try:
            with open(user_data_filename, 'r') as stream:
                try:
                    user_data = yaml.load(stream)
                    logger.info("USER DATA: " + str(user_data))
                    enc = EncryptionHandler(user_data['key_file'])
                    user = Client(user_data['uuid'], user_data['name'], (user_data['udp_ip'], user_data['udp_port']),
                                  enc.get_public_key(), None, enc=enc)
                    return user, enc
                except yaml.YAMLError as exc:
                    logger.debug(exc)
        except FileNotFoundError:
            logger.debug("User data .yml file not found, creating new user")
            return Client.create_user()
