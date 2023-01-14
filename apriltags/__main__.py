from typing import Final

import cv2
from numpy import typing as npt, int_
from pyapriltags import Detector

FAMILY: Final[str] = 'tag16h5'

def process_frame(img: cv2.Mat) -> cv2.Mat:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tags = Detector(
        searchpath=['apriltags'],
        families=FAMILY,
        nthreads=1,
        quad_decimate=5.0,
        quad_sigma=0.8,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    ).detect(
        gray, 
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None,
    )

    amt: int = 0
    for tag in tags:
        if tag.hamming > 0:
            continue
        amt += 1

        pts: npt.NDArray[int_] = tag.corners.astype(int)
        cv2.polylines(
            img, [pts],
            isClosed=True,
            color=(10, 255, 0),
            thickness=3,
        )

        x, y = pts[-1]
        cv2.putText(
            img, str(tag.tag_id),
            org=(x, y - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    print(f'tags: {amt}')
    return img

def test_sample() -> None:
    img = cv2.imread('tests/16h5_1.png')
    assert len(img)

    img = process_frame(img)
    cv2.imwrite('output/output.png', img)

def main() -> None:
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        assert len(frame)
        frame = process_frame(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()