import os
import glob


def check_dir():
    # source_dir = '/home/ubuntu/drive3/kinetics400_30fps_frames/train/*'
    source_dir = '/home/ubuntu/drive3/kinetics400_30fps_frames/val/*'

    classes = glob.glob(source_dir)

    # check videos number
    video_count = 0
    for class_idx in classes:
        videos = glob.glob(os.path.join(class_idx, '*'))
        video_count += len(videos)

    print("videos number: ", video_count)

    # check frames number
    frame_count = 0
    videos_not300_count = 0
    for class_idx in classes:
        videos = glob.glob(os.path.join(class_idx, '*'))
        for video in videos:
            frames = glob.glob(os.path.join(video, '*.jpg'))
            if len(frames) != 300:
                print("this video is not 300 frames: ",
                      video)
                videos_not300_count += 1
            else:
                frame_count += len(frames)
    print("frames number: ", frame_count)
    print("The number of videos which is not 300 frames: ", videos_not300_count)


def check_txt():
    # source_dir = '/home/ubuntu/drive3/kinetics400_30fps_frames/train/*'

    source_txt = '/home/ubuntu/drive3/kinetics400_30fps_frames/val.txt'
    data_split_path = '/home/ubuntu/drive3/kinetics400_30fps_frames/val'

    with open(source_txt) as txt:
        video_num = 0
        frame_num = 0
        video_not300frames = 0
        for filename in txt:
            video_num += 1

            video_name, _, label = filename.split()
            video_name, _ = os.path.splitext(video_name)
            video_name, _ = os.path.splitext(video_name)

            frames = sorted(glob.glob(os.path.join(data_split_path, video_name, "*.jpg")))
            if len(frames) != 300:
                print("this video is not 300 frames: ", video_name)
                video_not300frames += 1
            else:
                frame_num += len(frames)

        print("videos number: ", video_num)
        print("images number: ", frame_num)
        print("The number of videos which is not 300 frames: ", video_not300frames)

if __name__ == '__main__':
    check_txt()
