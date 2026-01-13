#!/usr/bin/env python3
import sys
import time
import argparse
from aragora.replay.storage import ReplayStorage
from aragora.replay.reader import ReplayReader


def list_cmd(args):
    storage = ReplayStorage()
    recordings = storage.list_recordings()
    print(f"Found {len(recordings)} recordings:")
    for r in recordings:
        print(f"{r['id']}: {r['topic']} ({r['status']})")


def replay_cmd(args):
    reader = ReplayReader(f".nomic/replays/{args.debate_id}")
    print(f"Replaying: {reader.meta.topic}")
    for event in reader.iter_events():
        print(f"[{event.offset_ms}ms] {event.source}: {event.content[:50]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    sub.add_parser("list").set_defaults(func=list_cmd)
    replay_p = sub.add_parser("replay")
    replay_p.add_argument("debate_id")
    replay_p.set_defaults(func=replay_cmd)
    args = parser.parse_args()
    args.func(args)
