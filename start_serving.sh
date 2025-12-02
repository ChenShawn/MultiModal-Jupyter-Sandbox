#!/bin/bash
set -x

# export TERM=dumb

mkdir -p uvilog
redis-server ./redis.conf >./uvilog/redis.log 2>&1 &
uvicorn fast_api_server_v2:app --host 0.0.0.0 --port 18901 --workers 8 >./uvilog/uv1.log 2>&1 &
# uvicorn fast_api_server:app --host 0.0.0.0 --port 18902 --workers 8 >./uvilog/uv2.log 2>&1 &
# uvicorn fast_api_server:app --host 0.0.0.0 --port 18903 --workers 8 >./uvilog/uv3.log 2>&1 &
# uvicorn fast_api_server:app --host 0.0.0.0 --port 18904 --workers 8 >./uvilog/uv4.log 2>&1 &

# nginx -c `pwd`/nginx.conf
wait
