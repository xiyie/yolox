# export PYTHONPATH=.

# python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 2 -o
# #11111111111git add .

# export PYTHONPATH=/project/train/src_repo/YOLOX/

# python /project/train/src_repo/YOLOX/tools/train.py -f /project/train/src_repo/YOLOX/exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 2 -o
#11111111111git add .
export PYTHONPATH=/project/train/src_repo/
# python /project/train/src_repo/tools/train.py -f /project/train/src_repo/exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 2 -o
python /project/train/src_repo/tools/train.py -f /project/train/src_repo/exps/example/yolox_voc/yolox_voc_tiny.py -d 0 -b 8 -o
