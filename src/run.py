import argparse

from werkzeug.middleware.profiler import ProfilerMiddleware

from app import app
from paths import (
    PROFILER_PATH
)

PORT = 8050


def main():
    parser = argparse.ArgumentParser(description="Run the application with or without profiling.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument("--prod", action="store_true", help="Start the app in production mode")
    args = parser.parse_args()

    if args.profile:
        run_profile_mode()
        return
    elif args.prod:
        run_prod_mode()
    else:
        app.run(debug=True, host='localhost', dev_tools_hot_reload_interval=1, port=PORT)


def run_profile_mode():
    # TODO Profiler should create a new dir with timestamp for the next log.
    app.server.config["PROFILE"] = True
    app.server.wsgi_app = ProfilerMiddleware(
        app.server.wsgi_app,
        sort_by=("cumtime", "tottime"),
        restrictions=[20],
        profile_dir=str(PROFILER_PATH)
    )
    print("Starting app in profiler mode")
    app.run(debug=True, host='localhost', dev_tools_hot_reload=False, port=PORT)


def run_prod_mode():
    import webbrowser
    from waitress import serve

    webbrowser.open(f"http://localhost:{PORT}/")
    serve(app.server, host='localhost', port=PORT)


if __name__ == "__main__":
    main()
