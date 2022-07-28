# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/200225 --output ../../ --confidence-threshold 0.05 --opts MODEL.WEIGHTS weights/eff_model_0504999.pth MODEL.RPN.POST_NMS_TOPK_TEST 1000

# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/200225 --output ../../ --confidence-threshold 0.5 --opts MODEL.WEIGHTS weights/eff_model_0384999.pth MODEL.RPN.POST_NMS_TOPK_TEST 300

# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/200225 --output ../../ --confidence-threshold 0.5 --opts MODEL.WEIGHTS weights/eff_model_0384999.pth MODEL.RPN.POST_NMS_TOPK_TEST 152



# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/191204_mobis --output ../../ --confidence-threshold 0.05 --opts MODEL.WEIGHTS weights/eff_model_0504999.pth MODEL.RPN.POST_NMS_TOPK_TEST 1000

# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/191204_mobis --output ../../ --confidence-threshold 0.5 --opts MODEL.WEIGHTS weights/eff_model_0384999.pth MODEL.RPN.POST_NMS_TOPK_TEST 300

# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/191204_mobis --output ../../ --confidence-threshold 0.5 --opts MODEL.WEIGHTS weights/eff_model_0384999.pth MODEL.RPN.POST_NMS_TOPK_TEST 152



CUDA_VISIBLE_DEVICES=7 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/svnetr3_zf --output ../../ --confidence-threshold 0.05 --opts MODEL.WEIGHTS weights/eff_model_0504999.pth MODEL.RPN.POST_NMS_TOPK_TEST 1000

# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/svnetr3_zf --output ../../ --confidence-threshold 0.5 --opts MODEL.WEIGHTS weights/eff_model_0384999.pth MODEL.RPN.POST_NMS_TOPK_TEST 300

# CUDA_VISIBLE_DEVICES=6 python demo/demo2.py --config-file configs/Base-RCNN-BiFPN-Eff.yaml --input /data/udb/alt_test/svnetr3_zf --output ../../ --confidence-threshold 0.5 --opts MODEL.WEIGHTS weights/eff_model_0384999.pth MODEL.RPN.POST_NMS_TOPK_TEST 152
