import logging
import time
from threading import Thread

import util.logging
from communication.chat import Chat
from communication.interface.intface import UserInterface
from message.socketmanager import RangeClientSocketManager
from packet import Port, PeerInitPacket, Address, TrackerPunchFromPacket, PeerMessagePacket
from peerdiscovery import TrackerPeerDiscoverer
from peerdiscovery.aggregatepeerdiscoverer import AggregatePeerDiscoverer
from peerdiscovery.localpeerdiscoverer import MultiLocalPeerDiscoverer
from util.client import Client

util.logging.setup()

logger = logging.getLogger()

MULTICAST_PORT = Port(38100)
MULTICAST_ADDRESSES = [("239.1.1.1", MULTICAST_PORT), ("224.0.0.1", MULTICAST_PORT)]

TRACKER_IP = "europium.sim642.eu"
TRACKER_PORT = Port(38000)
TRACKER_ADDRESS = (TRACKER_IP, TRACKER_PORT)

CLIENT_PORT_RANGE = range(38001, 38005 + 1)

user, enc = Client.get_user_data("config/user_data.yml")

sm = RangeClientSocketManager(CLIENT_PORT_RANGE)

multi_local_pd = MultiLocalPeerDiscoverer(user, sm, MULTICAST_ADDRESSES)
tracker_pd = TrackerPeerDiscoverer(user, sm, TRACKER_ADDRESS)
aggregate_pd = AggregatePeerDiscoverer([tracker_pd, multi_local_pd])  # local overrides tracker

logger.info(f"I am {user.uuid} aka {user.name}")
aggregate_pd.start(sm.dispatcher)

chats = {}

ui = UserInterface(user.name)


def ui_update_contacts():
    ui.update_contacts(aggregate_pd.get_clients([user.uuid]), chats)


def ensure_chat(peer: Client) -> Chat:
    if peer.uuid in chats:
        return chats[peer.uuid]
    else:
        chat = Chat(sm, peer, user)
        chat.start(sm.dispatcher)
        chat.update_contacts = ui_update_contacts
        chats[peer.uuid] = chat
        return chat


ui.ensure_chat = ensure_chat


def handle_peer_init(source: Address, packet: PeerInitPacket) -> None:
    # possibly unnecessary because unreliable if peer closes and reopens
    # done by handle_peer_message anyway
    ensure_chat(aggregate_pd.get_clients().get_peer_by_addr(source))


def handle_peer_message(source: Address, packet: PeerMessagePacket) -> None:
    peer = aggregate_pd.get_clients().get_peer_by_addr(source)
    if peer.uuid not in chats:
        chat = ensure_chat(peer)
        chat.handle_peer_message(source, packet)


def handle_tracker_punch_from(source: Address, packet: TrackerPunchFromPacket):
    sm.send_packet(PeerInitPacket(user.uuid, user.name, user.pubkey.exportKey("PEM")), packet.address)


sm.dispatcher.add_handler(PeerInitPacket, handle_peer_init)
sm.dispatcher.add_handler(PeerMessagePacket, handle_peer_message)
sm.dispatcher.add_handler(TrackerPunchFromPacket, handle_tracker_punch_from)

sm.start(daemon=True)


def ui_update():
    while True:
        ui_update_contacts()
        time.sleep(1)


ui_update_thread = Thread(target=ui_update, daemon=True)
ui_update_thread.start()

ui.run()
