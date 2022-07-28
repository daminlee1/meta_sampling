#python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/200225 --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
#mv /data/udb/alt/200225/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/200225

python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/gopro --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/gopro/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/gopro
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/gopro_kor --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/gopro_kor/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/gopro_kor
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/harman --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/harman/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/harman
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/hondari --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/hondari/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/hondari
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/movon_ger_od_tstld --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/movon_ger_od_tstld/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/movon_ger
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/movon_kor --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/movon_kor/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/movon_kor
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/movon_usa --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/movon_usa/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/movon_usa
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/skt --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/skt/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/skt
python tools/evaluate_xml.py --classset god_2020_10cls --data-root /data/udb/alt/youtube --images-dir JPEGImages --gt-xmls-dir Annotations --xmls-dir Anno_alt --iou-thres 0.8
mv /data/udb/alt/youtube/JPEGImages_dirty /data1/yjkim/alt_eval_output_img/youtube

