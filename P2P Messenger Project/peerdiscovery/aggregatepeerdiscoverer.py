from typing import List

from packet import PacketDispatcher
from util.peers import Peers
from .peerdiscoverer import PeerDiscoverer


class AggregatePeerDiscoverer(PeerDiscoverer):
	delegates: List[PeerDiscoverer]

	def __init__(self, delegates: List[PeerDiscoverer]) -> None:
		self.delegates = delegates

	def start(self, dispatcher: PacketDispatcher) -> None:
		for delegate in self.delegates:
			delegate.start(dispatcher)

	def announce(self) -> None:
		for delegate in self.delegates:
			delegate.announce()

	def get_clients(self, without=[]) -> Peers:
		clients = Peers()
		for delegate in self.delegates:
			for client in delegate.get_clients().get_client_list():
				if client.uuid not in without:
					clients.add_or_update_peer(client, update_active=False)
		return clients