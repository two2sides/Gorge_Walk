#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file tcp_util_test.py
# @brief
# @author kaiwu
# @date 2023-11-28


import queue
import time
import unittest
import argparse
from kaiwudrl.common.ipc.tcp_utils import TCPClient, TCPServer


def test_server():
    host = "127.0.0.1"
    port = 12345
    server = TCPServer(host, port)
    server.start()


def test_client():
    host = "127.0.0.1"  # 服务器的 IP 地址
    port = 12345  # 服务器的端口号
    client = TCPClient(host, port)
    client.connect()
    while True:
        client.send("Hello, Server!")
        data = client.receive()
        print(f"recv data is {data}")
    client.close()


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with command line arguments")
    parser.add_argument("--model", type=str, help="model, server or client")
    args = parser.parse_args()
    if args.model == "server":
        test_server()
    elif args.model == "client":
        test_client()
    else:
        print(f"not supported model {args.model}")

    unittest.main()
