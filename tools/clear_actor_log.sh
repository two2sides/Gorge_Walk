#!/bin/bash

# 清理actor机器上的日志目录

chmod +x tools/common.sh
. tools/common.sh

dir=/data/projects/gorge_walk/log/actor/
while true;
do
    # 注意删除的文件格式是以json和abs文件结尾的
    find "$dir" -type f \( -name "*.json" -o -name "*.log" \) -mmin +60 -exec rm -f {} \;
    sleep 600  # 等待10分钟
done