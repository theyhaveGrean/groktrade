"""Shared exceptions. Callers can catch :class:`AegisError` at the CLI boundary."""


class AegisError(Exception):
    """Base error for expected operational failures."""


class MissingCredentials(AegisError):
    """A required API key is not in the environment."""


class LiveTradingDisabled(AegisError):
    """This build refuses live capital. Paper mode is the only legal start state."""


class MassiveAPIError(AegisError):
    """Massive returned an error or the retry budget was exhausted."""


class PlanWindowError(AegisError):
    """The requested history window is empty after free-tier clamping."""


class HoldoutViolation(AegisError):
    """A research or optimization path tried to read the protected holdout."""


class RiskEngineNotInstalled(AegisError):
    """A strategy asked for the risk engine before the runner installed it."""
