import sys

from marsha.base import install_sigint_handler, run


def main() -> int:
    install_sigint_handler()
    try:
        return run()
    except KeyboardInterrupt:
        # Ctrl+C: quit promptly and cleanly instead of an asyncio traceback.
        print('\nInterrupted.', file=sys.stderr)
        return 130


# Entry point
sys.exit(main())
