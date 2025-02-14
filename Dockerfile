FROM ultralytics/ultralytics:8.3.40

RUN pip install -U numpy==1.26.4

COPY --chmod=754 yolo_video_inference.py ./

ENTRYPOINT ["./yolo_video_inference.py"]