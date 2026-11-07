import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..extensions import db
from ..localization.engine import LocalizationError, LocalizationUnavailable
from ..models import Localization
from .localization import create_record, execute_localization, save_upload, validate_image


class CopilotError(RuntimeError):
    pass


class CopilotService:
    def __init__(self, config, scene_manager, localization_engine):
        self.config = config
        self.scene_manager = scene_manager
        self.localization_engine = localization_engine

    @staticmethod
    def _public_scene(scene):
        return {
            key: scene[key]
            for key in (
                "id",
                "name",
                "ready",
                "has_gaussian_ply",
                "coordinate_frame_verified",
                "warning",
            )
        }

    def _facts(self):
        scenes = [self._public_scene(scene) for scene in self.scene_manager.list_scenes()]
        recent = [
            item.to_dict()
            for item in Localization.query.order_by(Localization.created_at.desc())
            .limit(10)
            .all()
        ]
        return {"scenes": scenes, "recent_localizations": recent}

    @staticmethod
    def _explicit_localization_request(messages):
        user_text = " ".join(
            item["content"]
            for item in messages
            if item.get("role") == "user" and isinstance(item.get("content"), str)
        )
        return bool(
            re.search(
                r"\b(?:locali[sz](?:e|er|ation)|run\s+(?:a\s+)?locali[sz]ation)\b",
                user_text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _tool_definitions(include_localize):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_scenes",
                    "description": "Retrieve the actual configured scenes and readiness.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_localizations",
                    "description": "Retrieve the latest localization results from the database.",
                    "parameters": {
                        "type": "object",
                        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_localization",
                    "description": "Retrieve one actual localization by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"localization_id": {"type": "string"}},
                        "required": ["localization_id"],
                    },
                },
            },
        ]
        if include_localize:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "run_localization",
                        "description": (
                            "Run localization on the image the user attached to this chat. "
                            "Only use after an explicit user request."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {"scene_id": {"type": "string"}},
                            "required": ["scene_id"],
                        },
                    },
                }
            )
        return tools

    def _execute_tool(self, name, arguments, attached_image):
        if name == "list_scenes":
            return [self._public_scene(scene) for scene in self.scene_manager.list_scenes()]
        if name == "list_localizations":
            limit = max(1, min(int(arguments.get("limit", 10)), 50))
            return [
                item.to_dict()
                for item in Localization.query.order_by(Localization.created_at.desc())
                .limit(limit)
                .all()
            ]
        if name == "get_localization":
            item = db.session.get(Localization, arguments["localization_id"])
            return item.to_dict() if item else {"error": "Localization not found."}
        if name == "run_localization":
            if attached_image is None:
                return {"error": "Attach an image before asking to run localization."}
            scene = self.scene_manager.get_scene(arguments["scene_id"])
            if scene is None:
                return {"error": "The requested scene is not configured."}
            image_bytes, filename = attached_image
            try:
                validate_image(image_bytes)
                image_path = save_upload(
                    image_bytes, filename, self.config["UPLOAD_DIR"]
                )
                record = create_record(filename, image_path, scene["id"])
                result = execute_localization(
                    record, scene, self.localization_engine
                )
                return {"localization": record.to_dict(), "result": result}
            except ValueError as exc:
                return {"error": str(exc)}
            except LocalizationUnavailable as exc:
                return {"localization": record.to_dict(), "error": str(exc)}
            except LocalizationError as exc:
                return {"localization": record.to_dict(), "error": str(exc)}
        return {"error": f"Unsupported tool: {name}"}

    def _fallback(self, messages, attached_image):
        facts = self._facts()
        scene_count = len(facts["scenes"])
        latest = facts["recent_localizations"][0] if facts["recent_localizations"] else None
        lines = [
            "AI Copilot is not configured: set LLM_API_KEY and restart the backend to enable generated responses.",
            f"Backend data is available: {scene_count} configured scene(s), "
            f"{Localization.query.count()} stored localization(s).",
        ]
        if latest:
            lines.append(
                "Latest stored result: "
                f"{latest['id']} — {latest['status']} — scene {latest['scene_id']}."
            )
        if attached_image and self._explicit_localization_request(messages):
            lines.append(
                "No localization was started because the AI tool-calling provider is not configured. "
                "Use the Localization page to run it directly."
            )
        return {"reply": "\n\n".join(lines), "provider_configured": False, "facts": facts}

    def _request_model(self, messages, tools):
        payload = {
            "model": self.config["LLM_MODEL"],
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        request = Request(
            f"{self.config['LLM_BASE_URL']}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config['LLM_API_KEY']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise CopilotError(f"LLM provider returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CopilotError(f"Could not reach the configured LLM provider: {exc}") from exc
        try:
            return result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CopilotError("LLM provider response did not contain a chat message.") from exc

    def chat(self, messages, attached_image=None, selected_scene_id=None):
        api_key = self.config["LLM_API_KEY"].strip()
        if not api_key:
            return self._fallback(messages, attached_image)
        facts = self._facts()
        explicit_request = self._explicit_localization_request(messages)
        can_localize = bool(attached_image and selected_scene_id and explicit_request)
        system_message = {
            "role": "system",
            "content": (
                "You are the AI Copilot for a visual localization research platform. "
                "Use only facts present in retrieved tool results or the supplied database snapshot; "
                "never invent poses, metrics, scene readiness, or scientific accuracy. "
                "Explain that translations are in COLMAP scene units unless proven metric. "
                "The notebook's depth-based 2D-3D route is experimental; this app's active route "
                "matches SIFT features to observed COLMAP 3D points and runs PnP-RANSAC. "
                "State uncertainty and failures explicitly. Run localization only when explicitly "
                "requested and an image is attached. Database snapshot: "
                + json.dumps(facts, ensure_ascii=False)
            ),
        }
        conversation = [system_message, *messages[-20:]]
        tools = self._tool_definitions(can_localize)
        for _ in range(4):
            message = self._request_model(conversation, tools)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return {
                    "reply": message.get("content") or "The AI provider returned an empty response.",
                    "provider_configured": True,
                    "facts": facts,
                }
            conversation.append(message)
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if name == "run_localization" and (
                    not can_localize or arguments.get("scene_id") != selected_scene_id
                ):
                    output = {"error": "Localization was not authorized or the scene did not match."}
                else:
                    output = self._execute_tool(name, arguments, attached_image)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )
        raise CopilotError("Copilot exceeded the maximum number of tool-call rounds.")
