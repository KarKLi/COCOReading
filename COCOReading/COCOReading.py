from pycocotools.coco import COCO
import numpy as np
import cv2 as cv
import random

dataDir='G:\\COCO2017'
dataType='val2017'
annFile='{}\\annotations_trainval2017\\annotations\\instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
category=dict()
for index,key in enumerate(cats):
    category[cats[index]['id']]=cats[index]['supercategory']
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

img_info=dict()
for key in coco.anns:
    bbox= coco.anns[key]['bbox']
    bbox = [int(i) for i in bbox]
    img_id=coco.anns[key]['image_id']
    filename=coco.imgs[img_id]['file_name']
    if filename in img_info:
        img_info[filename].append({category[coco.anns[key]['category_id']]:bbox})
    else:
        img_info[filename] = [{category[coco.anns[key]['category_id']]:bbox}]

for filename in img_info:
    img = cv.imread('%s\\%s\\%s\\%s'%(dataDir,dataType,dataType,filename))
    for List in img_info[filename]:
        for key,value in List.items():
            #if key != 'vehicle':
            #    break
            img = cv.rectangle(img,(value[0],value[1]),(value[0]+value[2],value[1]+value[3]),(0,0,255),1)
            img = cv.putText(img,key,(value[0],value[1]+value[3]),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv.imshow("Result",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
# 加载并显示图片,可以使用两种方式: 1) 加载本地图片, 2) 在线加载远程图片
# 1) 使用本地路径, 对应关键字 "file_name"
print('Done')