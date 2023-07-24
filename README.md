具体原文直接看主分支就好，本分支主要做了docker镜像打包以及docker-compose文件。直接复制docker-compose.yml到本地docker-compose up -d 即可。
需要预先安装nvidia-docker2以及docker-compose。
默认端口为7860，访问http://容器服务器IP:7860

如果网络环境好的话可以直接手动用dockerfile打包， 启动命令依然是docker run -itd 容器名:版本 
