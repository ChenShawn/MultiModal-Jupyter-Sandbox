## Release Logs

[20251202]
Update version Jupyter-Sandbox-V2, which is adapted from [GPTOSS tools](https://github.com/openai/gpt-oss/blob/main/gpt_oss/tools/python_docker/docker_tool.py).

The docker container can be found on dockerhub via:
```bash
docker pull chenshawn6915/multimodal-ipython-sandbox:oss-v2
```

----------------

## Introduction

Yet another code sandbox for agent RL training ... but in jupyter notebook style.
- Like jupyter notebook, user can execute multiple code blocks in the same session. All functions and variables are automatically stored after code execution.
- Like jupyter notebook, when calling `plt.show` in the code block, a base64 format image will be returned via `image` field.
- Like jupyter notebook, if the last line of code has no left value, the right value will be returned via `result` field.
- While there is no docker or any other vitualization, to ensure safety, all code executions are guarded by HumanEval safe prefix.

## Usage
### 1. Docker Virtualization
The docker image of this repo can be found on [this dockerhub repo](https://hub.docker.com/repository/docker/chenshawn6915/multimodal-ipython-sandbox/general).
```bash
docker pull chenshawn6915/multimodal-ipython-sandbox:latest
```

Execution:
```bash
docker run -d -p 18901-18904:18901-18904 chenshawn6915/multimodal-ipython-sandbox:latest
```

By default, there will be 4 code sandbox processes running on ports 18901-18904 respectively.
You can change the port mapping by modifying `-p 18901-18904:18901-18904` as you wish.

Testing:
```bash
python client_demo.py
```

### 2. Local Deployment
**NOTE:** Running this code sandbox on your local machine is **DANGEROUS**, as model-generated code execution can lead to unexpected behavior.
**DO NOT** do this unless for debugging purposes. We are not responsible for any harm that may result from your code generation.

Step 1: Install redis according to [their official installation documentation](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-from-source/)

> Make sure that redis-server and redis-cli are properly installed:
> which redis-server
> which redis-cli

Step 2: Start serving
```bash
bash start_serving.sh
```
