import logging
import socket
from threading import Thread
from typing import Tuple, Iterable

from packet import Packet, PacketDispatcher, Address, Port

logger = logging.getLogger(__name__)


class SocketManager:
    RECV_BUFSIZE = 4096

    def __init__(self, socket):
        self.socket = socket
        self.recv_thread = Thread(target=self.recv_loop, name="SocketManager recv")
        self.dispatcher = PacketDispatcher()

    def start(self, daemon=False):
        self.recv_thread.setDaemon(daemon)
        self.recv_thread.start()

    @property
    def local_address(self):
        return self.socket.getsockname()

    def send_packet(self, packet: Packet, address: Address):
        """
        Based on https://pymotw.com/3/socket/udp.html

        Open UDP socket and send it to the given address.

        :param message: message to send as str
        :param send_to: (IP, PORT)
        :return:
        """
        logger.debug(f"{self.local_address} --> {address}: {packet}")
        self.socket.sendto(packet.to_bytes(), address)

    def recv_loop(self):
        logger.info(f"Listening on {self.local_address}")

        while True:
            try:
                packet, source = self.recv_packet()
                self.dispatcher.dispatch(source, packet)
            except:
                logger.exception("Packet parsing failed.")

    def recv_packet(self) -> Tuple[Packet, Address]:
        bytes, source = self.socket.recvfrom(self.RECV_BUFSIZE)
        packet = Packet.from_bytes(bytes)
        logger.debug(f"{self.local_address} <-- {source}: {packet}")
        return packet, source


class ClientSocketManager(SocketManager):
    def __init__(self):
        sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        # Enable loop-back multi-cast - the local machine will also receive multicasts
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        # Binding to port 0 is the official documented way to bind to a OS-assigned random port.
        sock.bind(("0.0.0.0", 0))

        super().__init__(sock)


class RangeClientSocketManager(SocketManager):
    def __init__(self, port_range: Iterable[Port]):
        sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)

        bound = False
        for port in port_range:
            try:
                sock.bind(("0.0.0.0", port))
                bound = True
                break
            except OSError as e:
                logger.warning(e)

        if not bound:
            raise RuntimeError(f"Could not bind any of {port_range}")

        super().__init__(sock)


class MulticastSocketManager(SocketManager):
    def __init__(self, multicast_address: Address):
        msock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

        msock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            # Windows has no REUSEPORT, REUSEADDR is sufficient - https://stackoverflow.com/a/14388707
            msock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        msock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        msock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)

        msock.bind(("0.0.0.0", multicast_address[1]))
        msock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                         socket.inet_aton(multicast_address[0]) + socket.inet_aton("0.0.0.0"))

        super().__init__(msock)


class TrackerSocketManager(SocketManager):
    def __init__(self, tracker_port: Port):
        sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", tracker_port))

        super().__init__(sock)
