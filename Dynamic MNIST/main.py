import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

cnn = load_model('model.h5')


def draw_shape(event, x, y, flags, param):
    '''
    This is the mouse callback function that detects whenever
    the left mouse button is pressed, depressed and whenever
    the mouse is moved while the button is pressed. This function
    is to be bound to the window.
    '''

    global drawing, img, ix, iy
    # start drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # plot circles where the mouse goes
    elif event == cv.EVENT_MOUSEMOVE and drawing is True:
        # cv.circle(img, center=(x, y), radius=7, color=(128, 0, 255), thickness=-1)
        cv.line(img, (ix, iy), (x, y), color=(128, 0, 255), thickness=3)
        ix, iy = x, y

    # stop drawing
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False


def main():
    global drawing, img, ix, iy
    drawing = False     # True if mouse is pressed
    ix = iy = None

    # Create a black image and a window with the callback function binded to it
    img = np.zeros((600, 800, 3), np.uint8)
    cv.namedWindow('Canvas')
    cv.setMouseCallback('Canvas', draw_shape)

    # Press SPACE to analyse the drawing
    # Press ESC to quit the program
    print("Press SPACE to analyse the drawing.")
    print("Press ESC to quit.")
    while True:
        cv.imshow('Canvas', img)
        k = cv.waitKey(1) & 0xFF
        if k == 32:
            image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            image = cv.resize(image, dsize=(28, 28))
            image = np.reshape(image, (-1, 28, 28, 1))
            result = np.argmax(cnn.predict(image))
            print(result)
            print("Press any key to clear the canvas:")
            cv.waitKey(0)
            img = np.zeros((600, 800, 3), np.uint8)
        elif k == 27:
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
