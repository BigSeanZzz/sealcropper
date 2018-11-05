# sealcropper

### Sealcropper is a script based on opencv and python. You can crop seal to feed the machine learning algorithm.

#### Original images like this：
![book](81D85783FB9C4E629390B6E18346826D)

#### After cropping：


##### Usage：
###### 1.put your original images in folder  “seal”
###### 2.open file sealcropper.py and run
###### 3.get your cropped images in folder  “des”

##### Notice：
###### 1.In order to have better result，you might adjust the color filter in the code。
###### Choose one filter below or add your own filter：
##### # color filter
###### # 红色范围，数值按[b,g,r]排布，参考PS进行取值
###### # 橘红
###### # filter = [([0, 0, 200], [180, 180, 255])]
###### # 正红
###### # filter = [([0, 0, 120], [100, 100, 255])]
###### # 正红黄底
###### filter = [([0, 0, 120], [70, 70, 255])]
###### # 红棕
###### # filter = [([0, 0, 150], [120, 120, 255])]

##### 2.You might want to see the images every step，while adjusting the filter.  Set the log_output = 1:
###### # log_output = 1：打印过程，调试时用。
###### # log_output = 0：不打印过程，批量处理时用。
###### log_output = 1
###### You will see the processing like this:


