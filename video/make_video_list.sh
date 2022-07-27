
if [ -f '/video/video_list.txt' ];then
    echo "Remove video_list.txt"
    rm /video/video_list.txt
fi

for name in "${PWD}"/*; do
    str="$name"
    video_name=${str:7}
    
    case $video_name in
        *.mp4|*.mkv|*.avi|*.AVI|*.MP4|*.MKV)
            printf 'Save "%s"\n' "$video_name"
            echo $video_name >> video_list.txt
            ;;
        *)
    esac
done
echo "done make_video_list"
