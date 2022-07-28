used following pre-trained model from tensorflow model zoo: ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8

**my_ssd_mobnet**\
image data distribution:\
181 imgs in total\
86 imgs with WerkStueck class and/or Behaelter class visible,\
79 imgs with Hochregallager and WerkStueck class as wells as Behaelter class visible\
16 background imgs of Hochregallager, no labels included

test/train ratio: 23.2%, 76.8%


**my_ssd_mobnet_back_view**\
image data distribution:\
248 imgs in total\
86 imgs with WerkStueck class and/or Behaelter class visible,\
130 imgs with Hochregallager and WerkStueck class as wells as Behaelter class visible\
32 background imgs of Hochregallager, no labels included

test/train ratio: 20.6%, 79.4%