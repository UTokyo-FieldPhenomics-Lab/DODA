# Down COCO
COCO_Dir=datasets/coco
mkdir -p $COCO_Dir

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $COCO_Dir/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O $COCO_Dir/val2017.zip

unzip $COCO_Dir/annotations_trainval2017.zip -d $COCO_Dir
unzip $COCO_Dir/val2017.zip -d $COCO_Dir/images



# Download GWHD
GWHD_Dir=datasets
mkdir -p $GWHD_Dir

wget https://zenodo.org/records/5092309/files/gwhd_2021.zip -O $GWHD_Dir/gwhd_2021.zip
unzip $GWHD_Dir/gwhd_2021.zip -d $GWHD_Dir