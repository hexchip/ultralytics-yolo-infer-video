# ultralytics yolo infer video

基于[Ultralytics](https://github.com/ultralytics/ultralytics)镜像，通过设置[ENTRYPOINT](https://docs.docker.com/reference/dockerfile/#entrypoint)使其成为一个使用yolo模型推理视频文件的命令。

## 构建镜像

```bash
~/ultralytics-yolo-infer-video$ docker build -t hexchip/ultralytics-yolo-infer-video .
```

## 运行前准备

### 创建存放模型的目录

```bash
mkdir models
```

### 创建存放视频的目录

```bash
mkdir videos
```

## 运行参数

- **modelName:** 位于模型目录中的模型的文件名
- **videoName:** 位于视频目录中的视频的文件名

## 运行镜像

### WSL (Windows Subsystem for Linux)

```bash
modelName=best.pt videoName=test.mp4 && \
docker run -it --rm \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /mnt/wslg:/mnt/wslg \
-v /usr/lib/wsl:/usr/lib/wsl \
--device /dev/dxg \
--device /dev/dri/card0 \
--device /dev/dri/renderD128 \
-e DISPLAY=$DISPLAY \
-e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e PULSE_SERVER=$PULSE_SERVER \
--gpus all \
-v ./models:/ultralytics/models \
-v ./videos:/ultralytics/videos \
hexchip/ultralytics-yolo-infer-video \
--modelPath ./models/${modelName} \
--videoPath ./videos/${videoName}
```

Linux 和 Mac 系统暂时没有涉及，欢迎提交贡献