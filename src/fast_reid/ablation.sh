#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --eval-st-only MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.SPATIALTEMPORAL.ALPHA 1 TEST.SPATIALTEMPORAL.SMOOTH 50
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --eval-st-only MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.SPATIALTEMPORAL.ALPHA 5 TEST.SPATIALTEMPORAL.SMOOTH 50
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --eval-st-only MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.SPATIALTEMPORAL.ALPHA 10 TEST.SPATIALTEMPORAL.SMOOTH 50
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --eval-st-only MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.SPATIALTEMPORAL.ALPHA 20 TEST.SPATIALTEMPORAL.SMOOTH 50
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --eval-st-only MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.SPATIALTEMPORAL.ALPHA 5 TEST.SPATIALTEMPORAL.SMOOTH 10
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --eval-st-only MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.SPATIALTEMPORAL.ALPHA 5 TEST.SPATIALTEMPORAL.SMOOTH 25
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --eval-st-only MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth TEST.SPATIALTEMPORAL.ALPHA 5 TEST.SPATIALTEMPORAL.SMOOTH 100

