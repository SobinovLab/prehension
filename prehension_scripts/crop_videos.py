#!python3.11
import cv2
import numpy as np
import tqdm


def transform(ivp, ovp, frame_start, frame_end):
    # open reading file and get params
    cap = cv2.VideoCapture(ivp)
    img_size = [
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # where start
    cap.set(1, frame_start)

    # open video for writing
    out = cv2.VideoWriter(
        ovp, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        frame_rate, (img_size[0], img_size[1]))

    number_frames = frame_end - frame_start + 1

    for frame_num in tqdm.tqdm(range(number_frames)):
        ret, img = cap.read()
        img = img.astype(np.uint8)
        out.write(img)
    out.release()
    cap.release()


def main():
    in_video_paths = (
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam001\trial0\cam001.avi",
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam002\trial0\cam002.avi",
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam003\trial0\cam003.avi",
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam004\trial0\cam004.avi",
    )
    ou_video_paths = (
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam001\trial1\cam001.avi",
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam002\trial1\cam002.avi",
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam003\trial1\cam003.avi",
        r"S:\ProjectFolders\Prehension\Data\Tot_Miller\sessions\2024_01_19\cameras\cam004\trial1\cam004.avi",
    )

    frame_start = 290
    frame_end = 580

    for ivp, ovp in zip(in_video_paths, ou_video_paths):
        transform(ivp, ovp, frame_start, frame_end)


if __name__ == '__main__':
    main()
