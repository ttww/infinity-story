
import json
from app.models import SessionStatus, WorldState
from app.persistence.repositories import UserRepository, SessionRepository


class SessionManager:
    @staticmethod
    async def get_or_create_user(channel_user_id: str) -> str:
        user = await UserRepository.get_or_create(channel_user_id)
        return user["id"]

    @staticmethod
    async def get_or_create_session(user_id: str) -> dict:
        session = await SessionRepository.get_active_for_user(user_id)
        if session: return session
        return await SessionRepository.create(user_id)

    @staticmethod
    async def start_new_session(user_id: str) -> dict:
        return await SessionRepository.create(user_id)

    @staticmethod
    async def get_session(session_id: str) -> dict | None:
        return await SessionRepository.get(session_id)

    @staticmethod
    async def update_status(session_id: str, status: SessionStatus) -> dict | None:
        return await SessionRepository.update(session_id, status=status.value)

    @staticmethod
    async def set_scenario(session_id: str, scenario_id: str) -> dict | None:
        return await SessionRepository.update(session_id, scenario_id=scenario_id, status=SessionStatus.collecting_parameters.value)

    @staticmethod
    async def set_current_node(session_id: str, node_id: str) -> dict | None:
        return await SessionRepository.update(session_id, current_node_id=node_id)

    @staticmethod
    async def update_world_state(session_id: str, world_state: WorldState) -> dict | None:
        return await SessionRepository.update(session_id, world_state_json=world_state.model_dump_json())

    @staticmethod
    async def get_world_state(session: dict) -> WorldState:
        ws_json = session.get("world_state_json", "{}")
        data = json.loads(ws_json) if isinstance(ws_json, str) else ws_json
        return WorldState(**data)

    @staticmethod
    async def start_story(session_id: str, scenario_id: str, start_node_id: str) -> dict | None:
        return await SessionRepository.update(session_id, scenario_id=scenario_id, current_node_id=start_node_id, status=SessionStatus.running.value)

    @staticmethod
    async def pause_session(session_id: str) -> dict | None:
        return await SessionRepository.update(session_id, status=SessionStatus.paused.value)

    @staticmethod
    async def complete_session(session_id: str) -> dict | None:
        return await SessionRepository.update(session_id, status=SessionStatus.completed.value)

    @staticmethod
    async def add_message(session_id: str, direction: str, text: str) -> None:
        await SessionRepository.add_message(session_id, direction, text)

    @staticmethod
    async def get_messages(session_id: str, limit: int = 50) -> list[dict]:
        return await SessionRepository.get_messages(session_id, limit)

    @staticmethod
    async def save_generated_scene(session_id: str, node_id: str | None, scene_text: str, choices: list, state_updates: dict, llm_provider: str | None = None, token_usage: int = 0) -> None:
        await SessionRepository.save_scene(session_id, node_id, scene_text, choices, state_updates, llm_provider, token_usage)
