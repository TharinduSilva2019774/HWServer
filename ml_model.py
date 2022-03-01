import cv2

from mtcnn.mtcnn import MTCNN

from tensorflow.keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow
from keras import backend as K

test_dir = './NotCropedPhoto/temp.jpg'
test_processed_dir = './CropedPhotoTemp/'
BMI_model_name = "./models/BMI_f16.tflite"
AgeGender_model_name = "./models/AgeGender_fp16.tflite"
HeightWeight_model_name = "./models/height_weight_models.tflite"

detector = MTCNN()


def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def crop_img(im, x, y, w, h):
    return im[y:(y + h), x:(x + w), :]


def detect_face(face_path):
    img = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2RGB)
    box = detector.detect_faces(img)[0]
    return box


def save_image(imgData):
    print(imgData)


def crop_save_image():
    box = detect_face(test_dir)
    im = plt.imread(test_dir)
    cropped = crop_img(im, *box['box'])
    plt.imsave(test_processed_dir + "croped.jpg", crop_img(im, *box['box']))
    return "cropped"


def img2arr(img_path, version=1):
    """convert single image to array
    Args:
        @img_path: full path of the image (e.g. ./tmp/pic001.png)
    Return:
        np.array
    """
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = process_arr(img, version)
    return img


def process_arr(arr, version):
    """process array (resize, mean-substract)
    Args:
        @arr: np.array
    Return:
        np.array
    """
    img = cv2.resize(arr, (224, 224))
    img = np.expand_dims(img, 0)
    img = preprocess_input(img, version=version)
    return img


def load_model_BMI():
    BMI_interpreter_fp16 = tensorflow.lite.Interpreter(model_path=BMI_model_name)
    BMI_interpreter_fp16.allocate_tensors()
    print("BMI loaded")
    return BMI_interpreter_fp16


def load_model_AgeGender():
    AgeGender_interpreter_fp16 = tensorflow.lite.Interpreter(model_path=AgeGender_model_name)
    AgeGender_interpreter_fp16.allocate_tensors()
    print("age gender loaded")
    return AgeGender_interpreter_fp16


def load_model_HeightWeight():
    HeightWeight_interpreter_fp16 = tensorflow.lite.Interpreter(model_path=HeightWeight_model_name)
    HeightWeight_interpreter_fp16.allocate_tensors()
    print("height weight loaded")
    return HeightWeight_interpreter_fp16


def make_BMI_predictions(arr):
    BMI_interpreter_fp16 = load_model_BMI()

    input_index = BMI_interpreter_fp16.get_input_details()[0]["index"]
    output_index = BMI_interpreter_fp16.get_output_details()[0]["index"]

    BMI_interpreter_fp16.set_tensor(input_index, arr)
    BMI_interpreter_fp16.invoke()
    BMI_predictions = BMI_interpreter_fp16.get_tensor(output_index)

    print(float(BMI_predictions[0][0]))

    retults = [float(BMI_predictions[0][0])]
    print(type(retults))
    return retults


def make_AgeGender_predictions(arr):
    AgeGender_interpreter_fp16 = load_model_AgeGender()

    input_index = AgeGender_interpreter_fp16.get_input_details()[0]["index"]
    output_index1 = AgeGender_interpreter_fp16.get_output_details()[0]["index"]
    output_index2 = AgeGender_interpreter_fp16.get_output_details()[1]["index"]

    AgeGender_interpreter_fp16.set_tensor(input_index, arr)
    AgeGender_interpreter_fp16.invoke()
    gender = AgeGender_interpreter_fp16.get_tensor(output_index1)
    age = AgeGender_interpreter_fp16.get_tensor(output_index2)
    print(float(gender[0][0]))
    print(float(age[0][0]))

    retults = [float(age[0][0]), float(gender[0][0])]
    print(type(retults))
    return retults


def make_AgeGender_predictions(arr):
    AgeGender_interpreter_fp16 = load_model_AgeGender()

    input_index = AgeGender_interpreter_fp16.get_input_details()[0]["index"]
    output_index1 = AgeGender_interpreter_fp16.get_output_details()[0]["index"]
    output_index2 = AgeGender_interpreter_fp16.get_output_details()[1]["index"]

    AgeGender_interpreter_fp16.set_tensor(input_index, arr)
    AgeGender_interpreter_fp16.invoke()
    gender = AgeGender_interpreter_fp16.get_tensor(output_index1)
    age = AgeGender_interpreter_fp16.get_tensor(output_index2)
    print(float(gender[0][0]))
    print(float(age[0][0]))

    retults = [float(age[0][0]), float(gender[0][0])]
    print(type(retults))
    return retults


def make_HeightWeight_predictions(arr):
    HeightWeight_interpreter_fp16 = load_model_HeightWeight()

    input_index = HeightWeight_interpreter_fp16.get_input_details()[0]["index"]
    output_index1 = HeightWeight_interpreter_fp16.get_output_details()[0]["index"]
    output_index2 = HeightWeight_interpreter_fp16.get_output_details()[1]["index"]

    HeightWeight_interpreter_fp16.set_tensor(input_index, arr)
    HeightWeight_interpreter_fp16.invoke()
    height = HeightWeight_interpreter_fp16.get_tensor(output_index1)
    weight = HeightWeight_interpreter_fp16.get_tensor(output_index2)
    print(float(height[0][0]))
    print(float(weight[0][0]))

    retults = [float(height[0][0]), float(weight[0][0])]
    print(type(retults))
    return retults


def poundsToKG(pounds):
    return pounds * 0.45359237


def inchesToCm(inches):
    return inches * 2.54


def AGHWToBMR(age, gender, height, weight):
    height = inchesToCm(height)
    weight = poundsToKG(weight)
    if gender > 0.6:
        BMR = (10 * weight) + (6.25 * height) - (5 * int(age)) + 5
    else:
        BMR = (10 * weight) + (6.25 * height) - (5 * int(age)) - 161

    return BMR
