import logging
import os
import sys

import hydra
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

rich_theme = Theme(
    {
        "parsetree.spanlp": Style(color="white", bold=True),
        "parsetree.spanrp": Style(color="white", bold=True),
        "parsetree.spann1": Style(color="cyan", bold=True, italic=False),
        "parsetree.spann2": Style(color="cyan", bold=True, italic=False),
        "parsetree.bracket": Style(color="bright_black", bold=True),
    }
)


class CustomHighlighter(RegexHighlighter):
    base_style = "parsetree."
    highlights = [
        r"(?P<spanlp>\()(?P<spann1>\d+),\s*(?P<spann2>\d+)(?P<spanrp>\))",
        r"(?P<bracket>[\[\]])",
    ]


class CustomRichHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            rich_tracebacks=True,
            tracebacks_suppress=[hydra],
            log_time_format="[%X]",
            highlighter=CustomHighlighter(),
        )

        if os.environ.get("SLURM_JOB_ID") is not None:
            self.console.width = 140

    def get_level_text(self, record: logging.LogRecord) -> Text:
        """Get the level name from the record.

        Args:
            record (LogRecord): LogRecord instance.

        Returns:
            Text: A tuple of the style and level name.
        """
        level_name = record.levelname
        level_text = Text.styled(level_name[0], f"logging.level.{level_name.lower()}")
        return level_text
