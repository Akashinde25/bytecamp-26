"""
scenarios/run_demo.py

CLI demo runner for the multi-agent negotiation simulator.

Usage
-----
    python scenarios/run_demo.py --domain logistics
    python scenarios/run_demo.py --domain cloud
    python scenarios/run_demo.py --domain finance

The script starts a negotiation session directly (without the HTTP server)
by calling the core engine, so you can see GPT-4o negotiations in your
terminal within seconds of running the command.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import initialize_session, list_available_domains
from scenarios.reporter import (
    print_final_result,
    print_round_log,
    print_session_header,
)

logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO noise during demo
    format="%(levelname)s | %(name)s | %(message)s",
)


async def run_demo(domain: str) -> None:
    """Initialise and run a negotiation session, streaming output to terminal."""
    session_id = f"demo-{uuid.uuid4().hex[:8]}"
    print_session_header(session_id, domain)

    session = initialize_session(domain, session_id)

    # We'll collect messages by wiring the session queue and then calling run()
    queue: asyncio.Queue = asyncio.Queue()
    session._queue = queue

    # Run negotiation concurrent with consuming the queue
    async def consume() -> None:
        while True:
            msg = await queue.get()
            if msg.get("type") == "round":
                print_round_log(msg)
            elif msg.get("type") == "complete":
                print_final_result(msg["result"])
                break
            elif msg.get("type") == "error":
                print(f"\n[ERROR] {msg.get('message')}\n")
                break

    await asyncio.gather(session.run(), consume())


def main() -> None:
    available = list_available_domains()

    parser = argparse.ArgumentParser(
        description="Run a multi-agent negotiation demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available domains: {', '.join(available)}",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="logistics",
        choices=available if available else None,
        help="Domain to negotiate over (default: logistics).",
    )
    args = parser.parse_args()

    if args.domain not in available:
        print(
            f"[ERROR] Domain '{args.domain}' not found. "
            f"Available: {available}"
        )
        sys.exit(1)

    asyncio.run(run_demo(args.domain))


if __name__ == "__main__":
    main()
