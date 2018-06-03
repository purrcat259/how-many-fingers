import os
import shutil

import cv2
import time

ESCAPE_KEY = 27
SPACEBAR_KEY = 32

number_of_images_required_per_state = 150

if os.path.isdir('data'):
    print('Deleting old datasets')
    shutil.rmtree('data')
    time.sleep(1)
os.mkdir('data')

cam = cv2.VideoCapture(0)

# Six states, since 0, 1, 2, 3, 4, 5 fingers
for i in range(0, 6):
    # Create directory
    os.mkdir('data/{}'.format(i))
    state_images = []
    input('Press enter to start taking images')
    print('Taking images for state {}'.format(i))
    while True:
        ret_val, img = cam.read()
        cv2.imshow('Webcam for {} fingers up'.format(i), img)
        if cv2.waitKey(1) == SPACEBAR_KEY:
            state_images.append(img)
            print('{} images taken for state {}'.format(len(state_images), i))
            if len(state_images) > number_of_images_required_per_state:
                break
        if cv2.waitKey(1) == ESCAPE_KEY:
            break
    print('Closing OpenCV windows')
    cv2.destroyAllWindows()
    print('Saving state images')
    for image_count in range(0, len(state_images)):
        data_dir = 'data/{}'.format(i)
        print('Saving Image: {}'.format(image_count))
        cv2.imwrite(os.path.join(data_dir + '/{}.jpg'.format(image_count)), state_images[image_count])
