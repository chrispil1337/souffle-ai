# SPDX-License-Identifier: GPL-3.0-or-later
import os


def is_telemetry_enabled() -> bool:
    """Telemetry is opt-in in the Soufflé AI fork.

    Usage statistics are only sent when ``BUZZ_ENABLE_TELEMETRY`` is set.
    ``BUZZ_DISABLE_TELEMETRY`` always wins, so existing setups keep working.
    """
    if os.getenv("BUZZ_DISABLE_TELEMETRY"):
        return False
    return bool(os.getenv("BUZZ_ENABLE_TELEMETRY"))
