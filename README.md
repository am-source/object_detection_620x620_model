used following pre-trained model from [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md):
* ssd_mobilenet_v2_fpnlite_640x640
* ssd_resnet50_v1_fpn_640x640

###Custom trained models:

**my_ssd_mobnet_only_back_view**\
image data distribution:\
271 imgs in total\
86 imgs with WerkStueck class and/or Behaelter class visible,\
169 imgs with Hochregallager and WerkStueck class + Behaelter class visible\
32 background imgs of Hochregallager, no labels included\
Instances:\
test -   WerkStueck class 207, Behaelter 312\
train - WerkStueck class 711, Behaelter 932\
total - WerkStueck class 918, Behaelter 1044

test/train ratio: 20%, 80%


**my_ssd_mobnet_4batch_401img,**\
**my_ssd_resnet**\
image data distribution:\
401 imgs in total\
93 imgs with WerkStueck class and/or Behaelter class visible,\
276 imgs with Hochregallager and WerkStueck class + Behaelter class visible\
(166 back view, 110 front view of Hochregallager)\
32 background imgs of Hochregallager, no labels included\
Instances:\
test -   WerkStueck class 232, Behaelter 355\
train - WerkStueck class 1101, Behaelter 1539\
total - WerkStueck class 1333, Behaelter 1894

test/train ratio: 20%, 80%
