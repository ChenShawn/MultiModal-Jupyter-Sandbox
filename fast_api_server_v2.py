from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import io
import re
import asyncio
import base64
import argparse
import uvicorn
import time
from PIL import Image
import json

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from typing import Dict, Optional, Any, List

import redis
from redis_client import RedisClient
from local_jupyter_session import run_jupyter_code

app = FastAPI()
RC = RedisClient()
SEM = asyncio.Semaphore(4)


@app.post("/run_jupyter")
async def query(request: Request):
    input_data = await request.json()
    session_id = str(input_data.get("session_id", "JupyterSandboxDefault"))
    code_str = str(input_data.get("code", ""))
    if code_str.strip() == "":
        return {
            "output": "Error: No code provided",
            "status": "error",
            "execution_time": 0.0,
        }

    timeout = float(input_data.get("timeout", 5.0))

    start_time = time.time()
    async with SEM:
        jupyter_output = await run_jupyter_code(RC, session_id, code_str, timeout)
    end_time = time.time()
    execution_time = float(time.time() - start_time)

    response = {
        "output": jupyter_output,
        "status": "success",
        "execution_time": execution_time,
    }
    return response


@app.post("/clear_session")
async def jupyter_sandbox(request: Request):
    input_data = await request.json()
    session_id = str(input_data.get("session_id", "JupyterSandboxDefault"))
    RC.conn.expire(session_id, 1)
    return {"status": "success"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI server for code sandbox.")
    parser.add_argument("--port", type=int, default=12345, help="Port to run the FastAPI server on.")
    args = parser.parse_args()

    uvicorn.run("fast_api_server_v2:app", host="0.0.0.0", port=args.port, reload=False)