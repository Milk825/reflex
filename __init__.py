# Reflex - Self-correcting intelligence layer
# Root package exports

from .core.bridge import BridgeServer
from .core.integration import AutonomyCoordinator, ActionDirective
from .core.meta_cognition import MetaCognitionMonitor, LoopingState
from .core.preflight import PreflightChecker
from .core.rollout import RolloutController, RolloutMode
from .core.self_evaluation import SelfEvaluation
from .core.auto_reflection import AutoReflection
from .core.communication import CommunicationPolicy

__version__ = "0.1.0"

__all__ = [
    "BridgeServer",
    "AutonomyCoordinator",
    "ActionDirective",
    "MetaCognitionMonitor",
    "LoopingState",
    "PreflightChecker",
    "RolloutController",
    "RolloutMode",
    "SelfEvaluation",
    "AutoReflection",
    "CommunicationPolicy",
]