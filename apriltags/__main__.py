import cv2
from pyapriltags import Detector

def main():
    img = cv2.imread('tests/test2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = Detector(
        searchpath=['apriltags'],
        families='tag36h11',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    tags = detector.detect(
        gray, 
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None,
    )

    print(f'tags: {len(tags)}')
    for tag in tags:
        pts = tag.corners
        pts = pts.astype(int)

        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.imwrite('output/output.png', img)

if __name__ == '__main__':
    main()