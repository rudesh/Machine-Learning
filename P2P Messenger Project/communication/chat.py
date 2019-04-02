import logging
import uuid
from datetime import datetime
from typing import Callable, Any

from message.socketmanager import SocketManager
from packet import PeerInitPacket, PeerMessagePacket, Address, PacketDispatcher, PeerMessageOkPacket
from util.client import Client

logger = logging.getLogger()


class Message:
    owner: Client
    message_text: str
    received_time: datetime
    dispatch_time: datetime

    def __init__(self, owner: Client, message_text: str, received_time: datetime, dispatch_time: datetime):
        self.owner = owner
        self.message_text = message_text
        self.received_time = received_time
        self.dispatch_time = dispatch_time


class Chat:
    """
    This class instances hold relevant messaging data between two participants. For each conversation, a instance of
    this class is created.
    """
    sm: SocketManager
    peer: Client
    client: Client

    update_chat: Callable[[], Any]
    unread_count: int
    update_contacts: Callable[[], Any]

    def __init__(self, sm: SocketManager, peer: Client, client: Client):
        self.sm = sm
        self.peer = peer
        self.client = client
        self.msg_list = []
        self.update_chat = None
        self.unconfirmed = set()
        self.unread_count = 0
        self.update_contacts = None

    def start(self, dispatcher: PacketDispatcher) -> None:
        dispatcher.add_handler(PeerMessagePacket, self.handle_peer_message)
        dispatcher.add_handler(PeerMessageOkPacket, self.handle_peer_message_ok)
        self.peer.discoverer.start_chat(self.peer)
        self.sm.send_packet(PeerInitPacket(self.client.uuid, self.client.name, self.client.pubkey.exportKey("PEM")),
                            self.peer.address)

    def send_message(self, message: str) -> None:
        """
        Send message directly to other peer and keep track of it (if it gor received).
        """
        message_id = uuid.uuid4()
        iv, msg, aes_key = self.client.enc.encrypt_message(message, self.peer.pubkey)
        message_packet = PeerMessagePacket(message_id, datetime.now(), iv, msg, aes_key)

        self.sm.send_packet(message_packet, self.peer.address)
        # Keep track of messages that we send - rm them if they are received
        self.unconfirmed.add(message_id)
        self.msg_list.append(Message(self.client, message, message_packet.timestamp, message_packet.timestamp))
        self.update_ui()

    def handle_peer_message(self, source: Address, packet: PeerMessagePacket) -> None:
        """
        Handle incoming messages, e.g save them to list 'msg_list'
        """
        if source == self.peer.address:
            self.sm.send_packet(PeerMessageOkPacket(packet.message_id, datetime.now()), source)

            msg = self.client.enc.decrypt_message(packet.iv, packet.message, packet.aes_key)
            logger.info(msg)
            self.msg_list.append(Message(self.peer, msg, datetime.now(), packet.timestamp))
            self.update_ui()
            self.unread_count += 1
            if self.update_contacts is not None:
                self.update_contacts()

    def handle_peer_message_ok(self, source: Address, packet: PeerMessageOkPacket) -> None:
        """
        Keep track of the messages that this instance receives, e.g  update unconfirmed set.
        """
        if source == self.peer.address:
            if packet.message_id in self.unconfirmed:
                self.unconfirmed.remove(packet.message_id)

    def update_ui(self):
        if self.update_chat is not None:
            self.update_chat()  # tell UI
