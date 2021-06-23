import numpy as np
import os
import tensorflow as tf
import cv2

Notes = ['Eight', 'Sixteenth', 'Half','Whole','Quarter']
def predict(model):
    image_test = cv2.imread('../note.jpg')
    image_test = np.array([image_test])
    image_test = image_test /255
    return model.predict(image_test)

def viewPredict(pred):
    print('This is a ' + Notes[np.where(pred[0] == pred.max())[0].max()] + ' in ' + str(pred.max()*100) + '%')
def present(model):
    viewPredict(predict(model))

if __name__ == '__main__':
    print(tf.__version__)

    path = '../Notes/datasets/datasets/Notes/'
    label = os.listdir(path)

    try:
        label.remove('.DS_Store')  # potrzebne na systemie macos
    except:
        print('no file .DS_Store')

    mapping = {k: v for v, k in enumerate(label)}
    #   przygotowanie list zdjec i odpowiadajacych im labeli
    images=[]
    labels=[]

    #   odczyt zdjęć znajdujących się w katalogach
    print('Loading all images...')
    for i in label:
        listImages = os.listdir(path+i+'/')     # wyszukanie wszystkich zdjec w katalogu
        #print(i + ' = ' + str(mapping.get(i)) + ' -> ' + str(len(listImages)))
        print("All " + i + "are loded")
        for a in listImages:
            im = cv2.imread(path+i+'/'+a)   # odczyt zdjecia
            images.append(im)               # dodanie do listy zdjec
            labels.append(mapping.get(i))   # dodanie odpowiedniego labelu lecz zmapowanego na odpowiedni numerek [0,1,2,3,4]

    print('Reforming data...')
    #   rzutowanie na array numpy | mamy możliwość odczytu "kształtu" zapisanych danych
    labels = np.array(labels)
    images = np.array(images)
    images = images/255     # normalizacja pixeli

    #   podział danych na testowe i walidacyjne
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=.2)

    # WAŻNE - przekształcenie danych Y (czyli label zapisanyw formie [0-4]) na liste list 5-cio elementowych
    # dokłądniej [0] -> [1,0,0,0,0] | [3] -> [0,0,0,1,0]
    # potrzebne gdy mamy mieć kilka wyjść w sieci neuronoewj. Format danych do uczenia musi się zgadzać z danymi do sprawdzenia
    # w naszym przypadku jest 5 neuronów końcowych, tak potrzebujemy danych w formie 5-cio elementowej tablicy
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(y_train, len(label))
    y_test = to_categorical(y_test, len(label))

    print('Creating model...')
    # tworzenie modelu
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(len(label), activation='softmax'))

    model.summary()

    print('Compliling model...')
    # kompliacja modelu
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # liczba iteracji uczenia
    epochs = 15

    # uczenie sieci
    print('Training model...')
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

    # wykresy precyzji oraz strat
    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    fileName= 'model.h5'
    print('Saving model as '+fileName+' file')
    # zapis modelu do pliku
    model.save(fileName, save_format='h5')

    '''
    image = cv2.imread('../note.jpg')
    image = np.array([image])
    image = image /255
    pred = model.predict(image)
    
    
    pred = predict(model)
    viewPredict(pred)
    
    present(model)
    
    from keras.models import load_model
    model = load_model('model.h5')
    '''
