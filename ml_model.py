import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow
from keras import backend as K


def crop_img(im, x, y, w, h):
    return im[y:(y + h), x:(x + w), :]


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


def process_arr(arr, version):
    img = cv2.resize(arr, (224, 224))
    img = np.expand_dims(img, 0)
    img = preprocess_input(img, version=version)
    return img


class Asset:
    test_dir = './NotCropedPhoto/temp.jpg'
    test_processed_dir = './CropedPhotoTemp/croped.jpg'
    BMI_model_name = "./models/BMI_f16.tflite"
    AgeGender_model_name = "./models/AgeGender_fp16.tflite"
    HeightWeight_model_name = "./models/height_weight_models.tflite"

    def __init__(self,test_dir,test_processed_dir,BMI_model_name,AgeGender_model_name,HeightWeight_model_name):
        self.test_dir = test_dir
        self.test_processed_dir = test_processed_dir
        self.BMI_model_name = BMI_model_name
        self.AgeGender_model_name = AgeGender_model_name
        self.HeightWeight_model_name = HeightWeight_model_name

    detector = MTCNN()

    def detect_face(self):
        img = cv2.cvtColor(cv2.imread(self.test_dir), cv2.COLOR_BGR2RGB)
        box = self.detector.detect_faces(img)[0]
        return box

    def save_image(self, imgData):
        print(imgData)

    def crop_save_image(self):
        box = self.detect_face()
        im = plt.imread(self.test_dir)
        cropped = crop_img(im, *box['box'])
        plt.imsave(self.test_processed_dir, crop_img(im, *box['box']))


    def img2arr(self, img_path, version=1):
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

    def load_model_BMI(self):
        BMI_interpreter_fp16 = tensorflow.lite.Interpreter(model_path=self.BMI_model_name)
        BMI_interpreter_fp16.allocate_tensors()
        print("BMI loaded")
        return BMI_interpreter_fp16

    def load_model_AgeGender(self):
        AgeGender_interpreter_fp16 = tensorflow.lite.Interpreter(model_path=self.AgeGender_model_name)
        AgeGender_interpreter_fp16.allocate_tensors()
        print("age gender loaded")
        return AgeGender_interpreter_fp16

    def load_model_HeightWeight(self):
        HeightWeight_interpreter_fp16 = tensorflow.lite.Interpreter(model_path=self.HeightWeight_model_name)
        HeightWeight_interpreter_fp16.allocate_tensors()
        print("height weight loaded")
        return HeightWeight_interpreter_fp16

    def make_BMI_predictions(self, arr):
        BMI_interpreter_fp16 = self.load_model_BMI()

        input_index = BMI_interpreter_fp16.get_input_details()[0]["index"]
        output_index = BMI_interpreter_fp16.get_output_details()[0]["index"]

        BMI_interpreter_fp16.set_tensor(input_index, arr)
        BMI_interpreter_fp16.invoke()
        BMI_predictions = BMI_interpreter_fp16.get_tensor(output_index)

        print(float(BMI_predictions[0][0]))

        retults = [float(BMI_predictions[0][0])]
        print(type(retults))
        return retults

    def make_AgeGender_predictions(self, arr):
        AgeGender_interpreter_fp16 = self.load_model_AgeGender()

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

    def make_AgeGender_predictions(self, arr):
        AgeGender_interpreter_fp16 = self.load_model_AgeGender()

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

    def make_HeightWeight_predictions(self, arr):
        HeightWeight_interpreter_fp16 = self.load_model_HeightWeight()

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

    def poundsToKG(self, pounds):
        return pounds * 0.45359237

    def inchesToCm(self, inches):
        return inches * 2.54

    def AGHWToBMR(self, age, gender, height, weight):
        height = self.inchesToCm(height)
        weight = self.poundsToKG(weight)
        if gender > 0.6:
            BMR = (10 * weight) + (6.25 * height) - (5 * int(age)) + 5
        else:
            BMR = (10 * weight) + (6.25 * height) - (5 * int(age)) - 161

        return BMR
