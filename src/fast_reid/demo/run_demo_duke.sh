python demo/visualize_result.py --config-file configs/DukeMTMC/AGW_R50.yml \
--parallel --dataset-name 'DukeMTMC' --output logs/vis_duke_agw_R50_rerank \
--rank-sort none --label-sort descending --vis-label --num-vis 500 \
--opts MODEL.WEIGHTS logs/dukemtmc/agw_R50/model_final.pth TEST.RERANK.ENABLED True
