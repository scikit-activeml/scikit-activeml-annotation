import argparse

from werkzeug.middleware.profiler import ProfilerMiddleware

from app import app
from paths import (
    PROFILER_PATH
)


def main():
    parser = argparse.ArgumentParser(description="Run the application with or without profiling.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    args = parser.parse_args()

    if args.profile:
        # TODO Profiler should create a new dir with timestamp for the next log.
        app.server.config["PROFILE"] = True
        app.server.wsgi_app = ProfilerMiddleware(
            app.server.wsgi_app,
            sort_by=("cumtime", "tottime"),
            restrictions=[20],
            profile_dir=str(PROFILER_PATH)
        )
        print("Starting app in profiler mode")
        app.run(debug=True, dev_tools_hot_reload=False)
        return

    app.run(debug=True, dev_tools_hot_reload_interval=1)


if __name__ == "__main__":
    main()
