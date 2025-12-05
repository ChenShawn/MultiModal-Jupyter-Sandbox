import asyncio
import contextlib
import io
import os
import re
import json
import queue
import base64
import subprocess
from io import BytesIO
from PIL import Image
from pathlib import Path

def base64_to_image(base64_str: str) -> Image.Image:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_bytes))
        return img
    except Exception as err:
        return None


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    try:
        buffer = BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        img_bytes = buffer.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    except Exception as err:
        return None


def strip_ansi(s):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', s)


class LocalJupyterSession:
    """Stateful helper that proxies execution through a local Jupyter kernel."""

    def __init__(
        self,
        connection_file: str | None = None,
        *,
        timeout: float = 120.0,
    ) -> None:
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The dangerously_use_local_jupyter backend requires the jupyter_client package to be installed."
            ) from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(
                    f"Cannot find Jupyter connection file at '{connection_path}'."
                )
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            # Ensure the connection is ready before executing.
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            km = KernelManager()
            km.start_kernel()
            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
            self._km = km
            self._owns_kernel = True

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""

        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        image_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                stderr_parts.append("Timed out waiting for Jupyter kernel output.\n" + str(exc))
                break

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                text, img64, imgkey = None, None, None
                data = content.get("data", {})
                if "image/png" in data.keys():
                    imgkey = "image/png"
                    img64 = data["image/png"]
                elif "image/jpeg" in data.keys():
                    imgkey = "image/jpeg"
                    img64 = data["image/jpeg"]
                else:
                    text = data.get("text/plain")

                if img64 and imgkey:
                    # base64: "data:image/png;base64,xxx"
                    image_parts.append(f"data:{imgkey};base64,{img64}")
                elif text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain the shell channel to capture final execution status.
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                stderr_parts.append("Timed out waiting for Jupyter kernel execution reply.\n" + str(exec))
                break

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout_raw = "".join(stdout_parts)
        stderr_raw = "".join(stderr_parts)
        stdout = strip_ansi(stdout_raw)
        stderr = strip_ansi(stderr_raw)

        # NOTE: OpenAI GPT-OSS tools: DO NOT do this on the server side, let the client handle it.
        # if not stdout.strip():
        #     stdout = (
        #         "[WARN] No output available. Use print() to output anything to stdout to "
        #         "receive the output"
        #     )
        
        outjson = dict(
            stdout=stdout,
            stderr=stderr,
            stdout_raw=stdout_raw,
            stderr_raw=stderr_raw,
            images=image_parts,
        )
        return outjson

    def dump(self):
        from internal_states import dump_vars_func_string_v2 as dump_func
        dump_out = self.execute(dump_func, timeout=30)
        dump_out_json = json.loads(dump_out["stdout_raw"])
        return dump_out_json.get("base64", "")

    def load(self, state):
        from internal_states import load_vars_func_string_v2 as load_func
        return self.execute(
            load_func.format(internal_state_base64=state), 
            timeout=30
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        # self.close()
        pass


async def run_jupyter_code(client, session_id, code_string, timeout):
    jupyter_session = LocalJupyterSession(timeout=timeout)
    base64_data = client.conn.get(session_id)
    if base64_data:
        base64_data_string = base64_data.decode('utf-8')
        # print(f" [DEBUG 222] {base64_data_string}")
        ret = jupyter_session.load(base64_data_string)
    
    jupyter_output = jupyter_session.execute(code_string, timeout=timeout)
    state = jupyter_session.dump()
    client.conn.set(session_id, state)
    client.conn.expire(session_id, 3600 * 1)  # 1 hours expiration
    # print(f" [DEBUG 333] {state}")
    jupyter_session.close()
    return jupyter_output


def __test_run_jupyter_code(client, session_id, code_string, timeout):
    jupyter_session = LocalJupyterSession(timeout=timeout)
    base64_data = client.conn.get(session_id)
    if base64_data:
        base64_data_string = base64_data.decode('utf-8')
        print(f" [DEBUG 222] {base64_data_string}")
        ret = jupyter_session.load(base64_data_string)
        print(f" [DEBUG 444] {ret=}")
    
    jupyter_output = jupyter_session.execute(code_string, timeout=timeout)
    state = jupyter_session.dump()
    client.conn.set(session_id, state)
    client.conn.expire(session_id, 3600 * 1)  # 1 hours expiration
    print(f" [DEBUG 333] {state}")
    jupyter_session.close()
    return jupyter_output


if __name__ == "__main__":
    test_code_1 = """
import pandas as pd
import numpy as np
import os
import re

def foo():
    print("inside foo")

# from PIL import Image
# img = Image.open("./highlighted_space.jpg")

# while True:
#     time.sleep(1)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(np.arange(10), np.arange(10))
# plt.show()

a, b = 5, 10
c = [3,2,1]
print("hello world")
"""

    test_code_2 = """
carr = np.array(c)
print(f"{a=}, {b=}")
print(f"{carr=}")
foo()
    """

    # test_out_1 = jupyter_session.execute(test_code_1)
    # print(f"{test_out_1=}")
    # state = jupyter_session.dump()
    # jupyter_session.close()

    # jupyter_session = LocalJupyterSession(timeout=5)
    # jupyter_session.load(state)
    # test_out_2 = jupyter_session.execute(test_code_2)
    # print(f"{test_out_2=}")
    # jupyter_session.close()

    import uuid
    from redis_client import RedisClient
    RC = RedisClient()
    session_id = str(uuid.uuid4())
    out1 = __test_run_jupyter_code(RC, session_id, test_code_1, 10)
    print(f" [***] {out1=}")

    out2 = __test_run_jupyter_code(RC, session_id, test_code_2, 10)
    print(f" [***] {out2=}")
