![logo](images/logo.bmp)

[简体中文](README.md)|English



# EasyFace

A light weight face recognition project, high accuracy, real-time, cross-platform. Only need 410ms on Core i7 CPU. Face detection model RetinaFace's accuracy is 0.791 on WIDER Face Hard dataset. Face recognition MobileFacenet's accuracy is 99.55% on LFW dataset.

![result](images/result/img1.jpg)

## Characteristic

- Pure C++
- Cross-platform
- Easy to retrain face detection and recognition model
- Easy to integrate latest face detection and verification algrithm
- supported face detection: RetinaFace
- supported face verification: MobileFacenet

## Dependency

- [opencv3.4+](https://github.com/opencv/opencv)
- [ncnn](https://github.com/EasyFaceOrg/ncnn)

You should modify CMakeLists.txt according to the dependency's path

## Compile

### Linux

`./build_linux.sh`

### aarch64 Linux

`./build_aarch64-linux-gnu.sh`

## Run demo

`./run_demo.sh`



**result:**

```
Start face register... 
filename: images/register/huge.jpg
username: huge
filename: images/register/liuyifei.jpg
username: liuyifei
Finish face register. 
Start face identify 
identify result: liuyifei, 0.55
save result to: images/identify/liuyifei.result.jpg
blank frame grabbed
Finish face identify.
```



Total time is about 410ms on Core i7 CPU.

![time_comsume_cpu_i7.jpg](images/result/time_comsume_cpu_i7.jpg)



## demo result

registered face: huge, liuyifei

unregistered face: unknown

![cp1](images/result/cp1.jpg) ![cp2](images/result/cp2.jpg) ![cp3](images/result/cp3.jpg)



## Contact

Email：qianli_zh@qq.com

EasyFace QQ group：1070763980

![qqgroup](images/qqgroup.jpg)

## License

[BSD 3 Clause](LICENSE.txt)



## Acknowledge

[insightface](https://github.com/deepinsight/insightface)

[SeetaFace](https://github.com/seetafaceengine/SeetaFace2)

[ncnn](https://github.com/Tencent/ncnn)

[ncnn_example](https://github.com/MirrorYuChen/ncnn_example)

[opencv](https://github.com/opencv/opencv)