import logging
import time
from datetime import datetime
from threading import Thread
from typing import List

from Crypto.PublicKey import RSA

from message.socketmanager import SocketManager, MulticastSocketManager
from packet import PacketDispatcher, Address, LocalAnnouncePacket, LocalQueryPacket
from peerdiscovery.aggregatepeerdiscoverer import AggregatePeerDiscoverer
from util.client import Client
from util.peers import Peers
from .peerdiscoverer import ClientPeerDiscoverer

logger = logging.getLogger(__name__)


class LocalPeerDiscoverer(ClientPeerDiscoverer):
    TICKER_INTERVAL = 5

    def __init__(self, client: Client, socket_manager: SocketManager, multicast_address: Address) -> None:
        super().__init__(client)

        self.socket = socket_manager
        self.msocket = MulticastSocketManager(multicast_address)
        self.multicast_address = multicast_address

        self.clients = Peers()
        self.ticker_thread = Thread(target=self.ticker, daemon=True, name="LocalPeerDiscoverer ticker")

        self.tag = "L"

    def start(self, dispatcher: PacketDispatcher) -> None:
        self.msocket.dispatcher.add_handler(LocalAnnouncePacket, self.handle_local_announce)
        self.msocket.dispatcher.add_handler(LocalQueryPacket, self.handle_local_query)

        self.msocket.start(daemon=True)
        self.ticker_thread.start()

    def announce(self) -> None:
        packet = LocalAnnouncePacket(self.client.uuid, self.client.name, self.client.pubkey.exportKey('PEM'))
        self.socket.send_packet(packet, self.multicast_address)

    def get_clients(self) -> Peers:
        return self.clients

    def send_local_query(self):
        packet = LocalQueryPacket()
        self.socket.send_packet(packet, self.multicast_address)

    def handle_local_announce(self, source: Address, packet: LocalAnnouncePacket):
        client = Client(uuid=packet.client_id,
                        name=packet.name,
                        last_active=datetime.now(),
                        pubkey=RSA.importKey(packet.public_key),
                        address=source)
        client.discoverer = self
        self.clients.add_or_update_peer(client)

    def handle_local_query(self, source: Address, packet: LocalQueryPacket):
        self.announce()

    def ticker(self):
        self.send_local_query()

        while True:
            self.announce()

            time.sleep(self.TICKER_INTERVAL)


class MultiLocalPeerDiscoverer(AggregatePeerDiscoverer):
    def __init__(self, client: Client, socket_manager: SocketManager,
                 multicast_addresses: List[Address]) -> None:
        delegates = []
        for multicast_address in multicast_addresses:
            try:
                delegate = LocalPeerDiscoverer(client, socket_manager, multicast_address)
                delegates.append(delegate)
            except OSError as e:
                logger.warning(e)

        if not delegates:
            raise RuntimeError("No local peer discoverers")

        super().__init__(delegates)
