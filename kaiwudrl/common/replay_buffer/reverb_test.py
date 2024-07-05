#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file reverb_test.py
# @brief
# @author kaiwu
# @date 2023-11-28

import reverb

server = reverb.Server(
    tables=[
        reverb.Table(
            name="my_table",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=100,
            rate_limiter=reverb.rate_limiters.MinSize(1),
        ),
    ],
)

client = reverb.Client(f"localhost:{server.port}")
print(client.server_info())

client.insert([0, 1], priorities={"my_table": 1.0})

with client.trajectory_writer(num_keep_alive_refs=3) as writer:
    writer.append({"a": 2, "b": 12})
    writer.append({"a": 3, "b": 13})
    writer.append({"a": 4, "b": 14})

    # Create an item referencing all the data.
    writer.create_item(
        table="my_table",
        priority=1.0,
        trajectory={
            "a": writer.history["a"][:],
            "b": writer.history["b"][:],
        },
    )

    # Block until the item has been inserted and confirmed by the server.
    writer.flush()

print(list(client.sample("my_table", num_samples=2)))
