import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置采样数量
num = 300

# 设置采样间隔
frame_rate = 4

for i in range(num*frame_rate): 
    # 读取一帧数据
    ret, frame = cap.read()

    # 如果读取失败，退出循环
    if not ret:
        break

    # 显示视频画面
    cv2.imshow('frame', frame)

    # 每隔一定帧数保存一张图片
    if i % frame_rate == 0:
        num = num + 1
        cv2.imwrite("./collect/myface/20230427" + str(i).zfill(3) + ".jpg", frame)

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
