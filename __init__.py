"""MiniMax H3 Director SelfLift companion extension.

The original ComfyUI_MiniMaxH3_Director plugin supplies the shared Director
nodes, routes, and timeline UI. This companion registers only the independent
R2V SelfLift node so both plugins can stay installed together.
"""

from .nodes.director_selflift import MiniMaxH3DirectorSelfLift

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DirectorSelfLift": MiniMaxH3DirectorSelfLift,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DirectorSelfLift": "MiniMax H3 Director · SelfLift R2V",
}

WEB_DIRECTORY = "./web/selflift_js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
