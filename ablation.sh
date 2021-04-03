python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3_640vggbn/ --ngpu 4  --model 640_vggbn --batch_size 16 
CUDA_VISIBLE_DEVICES=3 python eval_refinedet_coco.py --prefix weights/align_4e3_640vggbn  --model 640_vggbn

python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3_640vggbn_wo_align/ --ngpu 4 --model 640_vggbn --batch_size 16  -woalign
CUDA_VISIBLE_DEVICES=3 python eval_refinedet_coco.py --prefix weights/align_4e3_640vggbn_wo_align  --model 640_vggbn -woalign

# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3_640vggbn_wo_align_refine/ --ngpu 4 --model 640_vggbn --batch_size 16  -worefine 
# CUDA_VISIBLE_DEVICES=3 python eval_refinedet_coco.py --prefix weights/align_4e3_640vggbn_wo_align_refine  --model 640_vggbn -worefine

# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3_640vggbn_wo_align_refine_fuse/ --ngpu 4 --model 640_vggbn --batch_size 16  -wofuse --resume ./weights/align_4e3_640vggbn_wo_align_refine_fuse/RefineDet640_COCO_epoches_280.pth --resume_epoch 280
# CUDA_VISIBLE_DEVICES=3 python eval_refinedet_coco.py --prefix weights/align_4e3_640vggbn_wo_align_refine_fuse  --model 640_vggbn  -wofuse

