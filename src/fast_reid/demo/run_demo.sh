python demo/visualize_result.py --config-file configs/Market1501/AGW_R50.yml \
--parallel --dataset-name 'Market1501' --output logs/vis_market_agw_R50_rerank \
--rank-sort none --label-sort descending --vis-label --num-vis 100 \
--opts MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.RERANK.ENABLED True
