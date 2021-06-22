import sys

import cv2
import numpy as np


Notes = ['Eight', 'Sixteenth', 'Half','Whole','Quarter']

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('To less parameters, try: PredictNote.py note.jpg')
        exit(0)

    from keras.models import load_model
    model = load_model('model.h5')

    try:
        im = cv2.imread(sys.argv[1])
    except:
        print("no file")

    #image = Image.fromarray(im, 'RGB')
    #image = image.resize((64,64))
    #image = np.array(image)
    image = np.array(im)
    image = image / 255
    pred = model.predict(np.asarray([image]))
    print('This is a ' + Notes[np.where(pred[0] == pred.max())[0].max()] + ' in ' + str(pred.max()) + '%')