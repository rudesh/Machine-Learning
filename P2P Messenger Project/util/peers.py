from typing import Dict, List

from Crypto.PublicKey import RSA

from packet import ClientID, Address, TrackerQueryReplyEntry
from util.client import Client


class Peers:
    def __init__(self):
        self.__addr_peers = {}  # type: Dict[Address, Client]
        self.__id_peers = {}  # type: Dict[ClientID, Client]

    def add_or_update_peer(self, client: Client, update_active=True):
        if update_active and client.uuid in self.__id_peers:
            client.set_last_active_time()
        self.__id_peers[client.uuid] = client
        self.__addr_peers[client.address] = client

    def add_tqre(self, discoverer, entry: TrackerQueryReplyEntry):
        client = Client(entry.client_id, entry.name, entry.address, RSA.importKey(entry.public_key), entry.timestamp)
        client.discoverer = discoverer
        self.add_or_update_peer(client, update_active=False)

    def add_tqres(self, discoverer, clients: List[TrackerQueryReplyEntry]):
        for client in clients:
            self.add_tqre(discoverer, client)

    def get_peer_by_addr(self, address: Address) -> Client:
        return self.__addr_peers[address]

    def get_peer_by_id(self, id: ClientID) -> Client:
        return self.__id_peers[id]

    def to_string(self) -> str:
        return "\n".join(list(map(lambda c: str(c), self.__id_peers.values())))

    def get_client_list(self) -> List[Client]:
        return list(self.__id_peers.values())

    def __str__(self):
        return "CLIENTS: \n" + "\n".join(str(c) for c in self.get_client_list())

    def to_tqres(self) -> List[TrackerQueryReplyEntry]:
        clients = []
        for client in self.__id_peers.values():
            clients.append(TrackerQueryReplyEntry(client.address, client.uuid, client.name, client.last_active, client.pubkey.exportKey("PEM")))
        return clients


