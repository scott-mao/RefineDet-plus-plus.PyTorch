############ voc 07 ##########

# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/voc_4e3_512vggbn/ --ngpu 4  --model 512_vggbn --batch_size 32 --dataset VOC -max 240
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/voc_4e3_512vggbn/ --ngpu 4  --model 512_vggbn --batch_size 32 --dataset VOC -max 240 --resume ./weights/voc_4e3_512vggbn/RefineDet512_VOC_epoches_120.pth --resume_epoch 120
# CUDA_VISIBLE_DEVICES=3 python eval_refinedet_voc07.py --prefix weights/voc_4e3_512vggbn  --model 512_vggbn 


python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/voc_4e3_512vgg/ --ngpu 4  --model 512_vggbn --batch_size 32 --dataset VOC -max 240 --pretrained --resume ./weights/voc_4e3_512vgg/RefineDet512_VOC_epoches_110.pth --resume_epoch 110
CUDA_VISIBLE_DEVICES=3 python eval_refinedet_voc07.py --prefix weights/voc_4e3_512vgg  --model 512_vggbn  -wobn
 