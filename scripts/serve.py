import os

import uvicorn


if __name__ == "__main__":
    host = os.environ.get("SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVICE_PORT", "8000"))
    uvicorn.run("access_perm.service:app", host=host, port=port)
