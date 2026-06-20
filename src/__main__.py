import sys

if len(sys.argv) > 1 and sys.argv[1] == "--cli":
    sys.argv.pop(1)
    from src.cli import main as cli_main

    cli_main()
else:
    from src.web import main as web_main

    web_main()
