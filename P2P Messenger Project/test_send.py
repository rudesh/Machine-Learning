import logging
import time
from threading import Thread
from uuid import UUID
from datetime import datetime

import util.logging
from communication.chat import Chat
from communication.interface.intface import UserInterface
from message.socketmanager import RangeClientSocketManager
from packet import Port, PeerInitPacket, Address, TrackerPunchFromPacket, PeerMessagePacket, PeerMessageOkPacket
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
aggregate_pd = AggregatePeerDiscoverer([tracker_pd, multi_local_pd]) # local overrides tracker

logger.info(f"I am {user.uuid} aka {user.name}")
aggregate_pd.start(sm.dispatcher)


ok_times = []
def handle_peer_message_ok(source: Address, packet: PeerMessageOkPacket) -> None:
    ok_times.append(datetime.now())

sm.dispatcher.add_handler(PeerMessageOkPacket, handle_peer_message_ok)

sm.start(daemon=True)

time.sleep(10)
# chat = Chat(sm, aggregate_pd.get_clients().get_peer_by_id(UUID("4ee5bf6d-bb12-4a6b-91a3-ee02fee6f989")), user) # Priit P
chat = Chat(sm, aggregate_pd.get_clients().get_peer_by_id(UUID("e7fd7b27-7a2b-4664-b091-c505a6b2099a")), user) # Mari

MSG_COUNT = 50
send_start = datetime.now()
for i in range(MSG_COUNT):
    chat.send_message("foobar" * 100)
send_end = datetime.now()

time.sleep(10)

print("Sending start:", send_start)
print("Sending end:", send_end)
print("Sending avg:", (send_end - send_start) / MSG_COUNT)
print("OK count:", len(ok_times))
ok_start = min(ok_times)
print("OK start:", ok_start)
ok_end = max(ok_times)
print("OK end:", ok_end)
print("OK avg:", (ok_end - ok_start) / MSG_COUNT)

# Mari:
# Sending start: 2018-12-03 09:54:08.104539
# Sending end: 2018-12-03 09:54:08.194867
# Sending avg: 0:00:00.001807
# OK count: 50
# OK start: 2018-12-03 09:54:08.128893
# OK end: 2018-12-03 09:54:13.710601
# OK avg: 0:00:00.111634

