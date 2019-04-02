import logging
import time
from threading import Thread

from message.socketmanager import SocketManager
from packet import TrackerAnnouncePacket, TrackerQueryPacket, TrackerAnnounceOkPacket, TrackerQueryReplyPacket, \
    PacketDispatcher, Address, TrackerPunchToPacket
from util.client import Client
from util.peers import Peers
from .peerdiscoverer import ClientPeerDiscoverer

logger = logging.getLogger(__name__)


class TrackerPeerDiscoverer(ClientPeerDiscoverer):
    clients: Peers

    TICKER_INTERVAL = 30

    def __init__(self, client: Client, socket: SocketManager, tracker_address: Address) -> None:
        super().__init__(client)

        self.socket = socket
        self.tracker_address = tracker_address

        self.clients = Peers()
        self.ticker_thread = Thread(target=self.ticker, daemon=True, name="TrackerPeerDiscoverer ticker")

        self.tag = "T"

    def start(self, dispatcher: PacketDispatcher) -> None:
        dispatcher.add_handler(TrackerQueryReplyPacket, self.handle_tracker_query_reply)
        dispatcher.add_handler(TrackerAnnounceOkPacket, self.handle_tracker_announce_ok)

        self.ticker_thread.start()

    def announce(self) -> None:
        packet = TrackerAnnouncePacket(self.client.uuid, self.client.name, self.client.pubkey.exportKey("PEM"))
        self.socket.send_packet(packet, self.tracker_address)

    def get_clients(self):
        return self.clients

    def start_chat(self, peer: Client) -> None:
        logger.info(f"Hole punching to {peer} via {self.tracker_address}")
        packet = TrackerPunchToPacket(peer.uuid)
        self.socket.send_packet(packet, self.tracker_address)

    def send_tracker_query(self):
        packet = TrackerQueryPacket()
        self.socket.send_packet(packet, self.tracker_address)

    def handle_tracker_announce_ok(self, source: Address, packet: TrackerAnnounceOkPacket):
        pass

    def handle_tracker_query_reply(self, source: Address, packet: TrackerQueryReplyPacket):
        self.clients.add_tqres(self, packet.clients)

    def ticker(self):
        while True:
            self.announce()
            self.send_tracker_query()

            time.sleep(self.TICKER_INTERVAL)
