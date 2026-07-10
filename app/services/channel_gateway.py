
from abc import ABC, abstractmethod
from typing import Any


class ChannelGateway(ABC):
    @abstractmethod
    async def receive_message(self, user_id: str, message: str) -> None: ...
    @abstractmethod
    async def send_message(self, user_id: str, message: str) -> None: ...
    @abstractmethod
    async def send_messages(self, user_id: str, messages: list[str]) -> None: ...


class MockWhatsAppGateway(ChannelGateway):
    def __init__(self):
        self._inbox: list[dict[str, Any]] = []
        self._outbox: list[dict[str, Any]] = []

    async def receive_message(self, user_id: str, message: str) -> None:
        self._inbox.append({"user_id": user_id, "message": message, "direction": "inbound"})

    async def send_message(self, user_id: str, message: str) -> None:
        self._outbox.append({"user_id": user_id, "message": message, "direction": "outbound"})

    async def send_messages(self, user_id: str, messages: list[str]) -> None:
        for msg in messages:
            await self.send_message(user_id, msg)

    def get_outbox(self) -> list[dict]: return list(self._outbox)
    def get_inbox(self) -> list[dict]: return list(self._inbox)
    def clear(self): self._inbox.clear(); self._outbox.clear()


class CLIDevGateway(ChannelGateway):
    async def receive_message(self, user_id: str, message: str) -> None: pass
    async def send_message(self, user_id: str, message: str) -> None:
        print(f"\n[Story -> {user_id}]\n{message}\n")
    async def send_messages(self, user_id: str, messages: list[str]) -> None:
        for msg in messages: await self.send_message(user_id, msg)


class RESTDevGateway(ChannelGateway):
    def __init__(self):
        self._pending: dict[str, list[str]] = {}
    async def receive_message(self, user_id: str, message: str) -> None: pass
    async def send_message(self, user_id: str, message: str) -> None:
        self._pending.setdefault(user_id, []).append(message)
    async def send_messages(self, user_id: str, messages: list[str]) -> None:
        self._pending.setdefault(user_id, []).extend(messages)
    def get_pending(self, user_id: str) -> list[str]: return self._pending.pop(user_id, [])
    def clear(self): self._pending.clear()


_gateways: dict[str, ChannelGateway] = {}

def get_gateway(channel: str = "whatsapp_mock") -> ChannelGateway:
    if channel not in _gateways:
        if channel == "whatsapp_mock": _gateways[channel] = MockWhatsAppGateway()
        elif channel == "cli_dev": _gateways[channel] = CLIDevGateway()
        elif channel == "rest_dev": _gateways[channel] = RESTDevGateway()
        else: raise ValueError(f"Unknown channel: {channel}")
    return _gateways[channel]
