"""Tests for CLI command aliases exposed in the main parser."""

from aragora.cli.parser import build_parser


def test_build_parser_accepts_debate_alias_for_ask():
    parser = build_parser()

    args = parser.parse_args(["debate", "Should we ship this change?"])

    assert args.command == "debate"
    assert args.task == "Should we ship this change?"
    assert args.func.__name__ == "cmd_ask"


def test_build_parser_help_lists_debate_alias_once():
    help_text = build_parser().format_help()

    assert "ask (debate)" in help_text
