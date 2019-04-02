import logging
from datetime import datetime

from Crypto.PublicKey import RSA

import util.logging
from message.socketmanager import TrackerSocketManager
from packet import TrackerQueryPacket, TrackerQueryReplyPacket, TrackerAnnouncePacket, TrackerAnnounceOkPacket, \
	TrackerPunchToPacket, TrackerPunchFromPacket, Address, Port
from util.client import Client
from util.peers import Peers

util.logging.setup()

logger = logging.getLogger()

TRACKER_PORT = Port(38000)

sm = TrackerSocketManager(TRACKER_PORT)

clients = Peers()


def handle_tracker_query(source: Address, packet: TrackerQueryPacket):
	reply_packet = TrackerQueryReplyPacket(clients.to_tqres())
	sm.send_packet(reply_packet, source)


def handle_tracker_announce(source: Address, packet: TrackerAnnouncePacket):
	entry = Client(packet.client_id, packet.name, source, RSA.importKey(packet.public_key), datetime.now())
	clients.add_or_update_peer(entry)

	logger.debug("Clients:")
	logger.debug(clients.to_string())

	reply_packet = TrackerAnnounceOkPacket(datetime.now())
	sm.send_packet(reply_packet, source)


def handle_tracker_punch_to(source: Address, packet: TrackerPunchToPacket):
	entry = clients.get_peer_by_addr(source)
	to_entry = clients.get_peer_by_id(packet.client_id)
	packet = TrackerPunchFromPacket(source, entry.uuid, entry.name)
	sm.send_packet(packet, to_entry.address)


sm.dispatcher.add_handler(TrackerQueryPacket, handle_tracker_query)
sm.dispatcher.add_handler(TrackerAnnouncePacket, handle_tracker_announce)
sm.dispatcher.add_handler(TrackerPunchToPacket, handle_tracker_punch_to)

sm.start()
