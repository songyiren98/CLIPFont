CLIPVG使用说明文档


一、环境配置
    1. clip 按照官方文档安装
    2. diffvg 按照官方文档安装、编译，注意python-opencv版本需保持一致
    3. pytorch

```bash
apt-get update
apt-get install ffmpeg
pip install -r requirements.txt
```

二、路径准备
```bash
1. cd ../diffvg/app/（diffvg的安装路径）
2. 将本项目imgs文件夹和代码(CLIPVG_main.py, utils.py等)放置在app/下，如../diffvg/app/CLIPVG_main.py
3. 将使用的初始化矢量图和需拟合的栅格图放到 ../diffvg/app/imgs/中（已包含运行demo所需的输入）
```

三、运行
```bash
1. Peter人脸 python CLIPVG_main.py imgs/facenewlive2.svg imgs/face2.jpeg（facenewlive2.svg是输入的矢量图，face2.jpeg是需拟合的栅格图，仅在需要L2loss约束时会用到）
2. 半身角色生成 python clipvg_halfbody.py imgs/bodyx-c100.svg imgs/face2.jpeg
3. 字体风格化风格化 python clipvg_font.py imgs/ABCD.svg imgs/face2.jpeg
4. 建筑生成 python clipvg_baseline.py --svg imgs/cube2__2_.svg 
5. 车辆生成 python clipvg_baseline.py --svg imgs/car3__3_.svg 
```   

四、高级功能
```bash
1. 形状锁，颜色锁(全局锁定，将颜色或形状对应lr置为0)，用mask锁定特定区域，以卡通头像表情编辑为例，锁定面部以外区域  python clipvg_masklock.py imgs/cartoonface03.svg imgs/mycartoonface.png
2. 从0开始层次化生成,随机初始化50个形状。 python clipvg_G1.py  读取结果(results/body/result1.svg)，在上一步的基础上生成细节 python clipvg_G2.py --trans 3 --num 150 --radius 0.25 (num是新加的多边形数目；adius是新加的多边形尺寸大小，一般取0.3-0.1，随优化阶段逐步减小；trans是透明度，取1-10，真实透明度范围为0～1/trans的uniform distribution)。继续叠加细节，python clipvg_G2.py --trans 3.5 --num 200 --radius 0.2 
3. 加相似性Loss约束，以卡通头像表情编辑为例 python clipvg_similarloss.py imgs/cartoonface03.svg imgs/mycartoonface.png
```   

五、参数详解
```bash
1. 文本描述，修改代码中的prompts
2. --num_iter default= 150 （优化轮数，一般100-300）
3. points_vars, lr= 0.2 形状优化学习率（人脸一般为1.0-0.2，越大变化越剧烈，设为0则不优化形状）
4. color_vars, lr=0.01 颜色优化学习率 （一般为0.1-0.01，越大变化越剧烈，设为0则不优化颜色）
5. --lambda_patch，default=50 局部采样算loss的权重
6. --lambda_dir，default=100 全局采样算loss的权重
7. --crop_size， default = 350 局部采用尺寸
8. --num_crops，default = 64 局部采样数目
9. --img_size，default = 512 输入图像/矢量图默认大小为512
```   

六、常见问题及解决方案
```bash
1. 运行报错core—dump…… 原因：1. python-opencv版本不对 2. svg初始化有问题，包含非曲边多边形如圆，长方形，非封闭路径（在AI里删除，编辑形状没删干净）。
2. 运行代码结果和论文/举例的不同。原因：参数设置没有完全一致，需理解每个参数的含义及对最终结果的影响从而灵活调整，获得任意想要的效果。
3. 用mask锁定形状变化报错，索引超出范围，因为svg文件中形状超出边界，用Adobe AI转不会出现问题，LIVE和diffvg要检查形状是否边界。
``` 
