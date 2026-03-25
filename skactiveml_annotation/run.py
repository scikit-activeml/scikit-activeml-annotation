from argparse import ArgumentParser
import logging

from dash import Dash

from skactiveml_annotation import util
from skactiveml_annotation.app import create_app
import skactiveml_annotation.paths as sap


PORT = 8050


def main():
    parser = ArgumentParser(description="Run the application with or without profiling.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument("--prod", action="store_true", help="Start the app in production mode")
    args = parser.parse_args()

    app = create_app()

    if args.profile:
        run_profile_mode(app)
        return
    elif args.prod:
        run_prod_mode(app)
    else:
        run_debug_mode(app)


def run_debug_mode(app: Dash):
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == 'true':
        util.logging.setup_logging()

    app.run(debug=True, host='localhost', dev_tools_hot_reload_interval=1, port=PORT)

def run_profile_mode(app: Dash):
    util.logging.setup_logging()

    try:
        from werkzeug.middleware.profiler import ProfilerMiddleware
    except ImportError as e:
        logging.error(f'Cannot run in profile mode because \'werkzeug\' is not installed: {e}')
        import sys
        sys.exit(1)

    app.server.config["PROFILE"] = True
    app.server.wsgi_app = ProfilerMiddleware(
        app.server.wsgi_app,
        sort_by=("cumtime", "tottime"),
        restrictions=[20],
        profile_dir=str(sap.PROFILER_PATH)
    )
    logging.info("Starting app in profiler mode")
    app.run(debug=True, host='localhost', dev_tools_hot_reload=False, port=PORT)


def run_prod_mode(app: Dash):
    util.logging.setup_logging()
    import webbrowser
    try:
        from waitress import serve  # pyright: ignore[reportMissingModuleSource]
    except ImportError as e:
        logging.error(e)
        import sys
        sys.exit(1)

    webbrowser.open(f"http://localhost:{PORT}/")
    serve(app.server, host='localhost', port=PORT)


if __name__ == "__main__":
    main()
