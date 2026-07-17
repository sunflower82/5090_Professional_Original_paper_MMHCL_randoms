"""
logging.py — Dual-Destination Logger
======================================

Provides a simple Logger class that writes timestamped messages to:
  1. The console (stdout).
  2. A per-experiment log file (in the run-specific directory).
  3. Optionally, a second log file in a shared aggregation directory
     (useful for comparing results across ablation experiments).

File structure created by the logger:
    ../<dataset>/<path_name>/<path_name>.txt     — per-run detailed log
    ../<dataset>/MM/<path_name>.txt              — copy of the same log
    ../<dataset>/MM/sum_<ablation>.txt           — one-line summaries
"""

from __future__ import annotations

import os
from datetime import datetime
import pathlib
from typing import Optional


class Logger:
    """
    Timestamped file + console logger.

    Attributes:
        target          : base name for the log file (without .txt)
        path            : directory for the primary log file
        log_            : whether file logging is enabled
        path2           : optional secondary directory for log copies
        ablation_target : tag for the ablation summary file
    """

    def __init__(
        self,
        path: str,
        is_debug: str,
        target: str = "log",
        path2: Optional[str] = None,
        ablation_target: Optional[str] = None,
    ) -> None:
        # Ensure the log directory exists
        pathlib.Path(f"{path}").mkdir(parents=True, exist_ok=True)
        self.target: str = target
        self.path: str = path
        self.log_: str = is_debug                        # file logging enabled?
        self.path2: Optional[str] = path2                # secondary (shared) log directory
        self.ablation_target: Optional[str] = ablation_target

        # Mark the start of a new training run in the log
        self.logging("#" * 30 + "   New Logger Start   " + "#" * 30)

    def logging(self, s: object) -> None:
        """
        Log a message with a timestamp to console and (optionally) to file(s).

        Output format:
            2026-02-09-14:30: <message>

        If ``is_debug`` is truthy, the message is also appended to:
            <path>/<target>.txt         — primary log file
            <path2>/<target>.txt        — secondary log file (if path2 is set)
        """
        s: str = str(s)
        # Always print to console
        print(datetime.now().strftime('%Y-%m-%d-%H:%M:'), s)

        # Append to primary log file
        if self.log_:
            with open(os.path.join(self.path, f"{self.target}.txt"), 'a+') as f_log:
                f_log.write(
                    str(datetime.now().strftime('%Y-%m-%d %H:%M:')) + s + '\n'
                )
            # Append to secondary (shared) log file for cross-run comparison
            if self.path2:
                with open(os.path.join(self.path2, f"{self.target}.txt"), 'a+') as f_log:
                    f_log.write(
                        str(datetime.now().strftime('%Y-%m-%d %H:%M:')) + s + '\n'
                    )

    def logging_sum(self, s: str) -> None:
        """
        Write a one-line summary to the shared ablation summary file.

        File: <path2>/sum_<ablation_target>.txt

        This is called once at the end of training with the final test
        results, making it easy to compare outcomes across different
        ablation configurations or random seeds.
        """
        if self.path2:
            print(s)
            with open(
                os.path.join(self.path2, f"sum_{str(self.ablation_target)}.txt"),
                'a+'
            ) as f_log:
                f_log.write(s + '\n')
