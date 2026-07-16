"""Launch the real server on a temp data dir for the Playwright smoke."""

import pathlib
import sys
import tempfile

import uvicorn

from planefuse_server.main import create_app

app = create_app(pathlib.Path(tempfile.mkdtemp(prefix="fs-e2e-")))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
