# 机器学习人脸识别
face_recognition.py------是本次项目python文件

paddle-------------------是paddle网站的备份以及下载文件

data---------------------是项目数据数据集文件

result-------------------是编译结果

collect------------------是用于收集人脸的文件

## 运行环境
本地环境：
windows10--------python3.8----------paddle

本次工程使用 CPU 训练。

pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

如果使用GPU,需要安装paddleGPU相关的包

pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple


## 网页端识别人脸：

问题：网页端需要实现识别自己人脸。但是文件上传不方便，且不安全。

思路:本地采集自己的人脸数据图片，放入本地数据集进行训练，最后在网页端上传模型权重文件，在网页端直接进行预测。


### 人脸数据集采集
1. 打开 collect 文件夹下的 collect.py 直接运行。
   
2. 采集后的人脸图片位于 ./collect/myface 文件下面。

3. 将myface文件夹复制到 ./data/data2393/images/face 目录下 

4. 本次我使用vgg网络，因此修改模型输出四个类别，在 face_recognition.py 代码中277行 ，将 predict = vgg_bn_drop(image=image, type_size=3) ，修改为 predict = vgg_bn_drop(image=image, type_size=4)

5. 训练模型，程序会自动将人脸 myface 加入到列表中（人脸文件夹的名字就是类别名）
   
6. 修改标签位置，在 face_recognition.py 代码414行label_list中，将标签顺序与
 ./data/data2393/face/readme.json 中标签对应

7. 预测

### 运行结果:

<left><img src="./data/data22141/test00002.jpg" width = 20%><left>

results: 0

infer results: jiangwen