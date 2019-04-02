from packet import PacketDispatcher
from util.client import Client
from util.peers import Peers


class PeerDiscoverer:
    def start(self, dispatcher: PacketDispatcher) -> None:
        raise NotImplementedError()

    def announce(self) -> None:
        raise NotImplementedError()

    def get_clients(self) -> Peers:
        raise NotImplementedError()

    def start_chat(self, peer: Client) -> None:
        pass


class ClientPeerDiscoverer(PeerDiscoverer):
    client: Client

    def __init__(self, client: Client) -> None:
        super().__init__()

        self.client = client
