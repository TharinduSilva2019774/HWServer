from unittest import TestCase
from ml_model import Asset
from ml_model import process_arr
import os
from tensorflow.keras.preprocessing import image

Assets = Asset('./website/NotCropedPhoto/tempForTest.jpg',
               './website/CropedPhotoTemp/cropedForTest.jpg',
               "./website/models/BMI_f16.tflite",
               "./website/models/AgeGender_fp16.tflite",
               "./website/models/height_weight_models.tflite")

class TestAsset(TestCase):
    # Test if face detect function work
    def test_detect_face(self):
        try:
            Assets.detect_face()
            self.assertTrue(True)
        except:
            self.fail()

    # Test if crop and save function work
    def test_crop_save_image(self):
        if os.path.exists(Assets.test_processed_dir):
            os.remove(Assets.test_processed_dir)

        Assets.crop_save_image()

        if os.path.exists(Assets.test_processed_dir):
            self.assertTrue(True)
        else:
            self.fail()

    # Test if Image to array converter works
    def test_img2arr(self):
        img = image.load_img('./website/CropedPhotoTemp/cropedForTest.jpg')
        img = image.img_to_array(img)
        img = process_arr(img, 1)
        self.assertEqual(img.all(), Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1).all())

    # Test if BMI model loader function works
    def test_load_model_bmi(self):
        BMI_interpreter_fp16 = Assets.load_model_BMI()
        arr = Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1)
        input_index = BMI_interpreter_fp16.get_input_details()[0]["index"]
        output_index = BMI_interpreter_fp16.get_output_details()[0]["index"]

        BMI_interpreter_fp16.set_tensor(input_index, arr)
        BMI_interpreter_fp16.invoke()
        BMI_predictions = BMI_interpreter_fp16.get_tensor(output_index)
        self.assertEqual(int(BMI_predictions), 24)

    # Test if age and gender model loader function works
    def test_load_model_age_gender(self):
        AgeGender_interpreter_fp16 = Assets.load_model_AgeGender()
        arr = Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1)
        input_index = AgeGender_interpreter_fp16.get_input_details()[0]["index"]
        output_index1 = AgeGender_interpreter_fp16.get_output_details()[0]["index"]
        output_index2 = AgeGender_interpreter_fp16.get_output_details()[1]["index"]

        AgeGender_interpreter_fp16.set_tensor(input_index, arr)
        AgeGender_interpreter_fp16.invoke()
        gender = AgeGender_interpreter_fp16.get_tensor(output_index1)
        age = AgeGender_interpreter_fp16.get_tensor(output_index2)
        if int(gender) == 0 and int(age) == 13:
            self.assertTrue(True)
        else:
            self.fail("gender or age !=")

    # Test if height and weight model loader function works
    def test_load_model_height_weight(self):

        HeightWeight_interpreter_fp16 = Assets.load_model_HeightWeight()
        arr = Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1)
        input_index = HeightWeight_interpreter_fp16.get_input_details()[0]["index"]
        output_index1 = HeightWeight_interpreter_fp16.get_output_details()[0]["index"]
        output_index2 = HeightWeight_interpreter_fp16.get_output_details()[1]["index"]

        HeightWeight_interpreter_fp16.set_tensor(input_index, arr)
        HeightWeight_interpreter_fp16.invoke()
        height = HeightWeight_interpreter_fp16.get_tensor(output_index1)
        weight = HeightWeight_interpreter_fp16.get_tensor(output_index2)
        if int(height) == 69 and int(weight) == 167:
            self.assertTrue(True)
        else:
            self.fail("height or weight !=")

    # Test if BMI predication function works
    def test_make_bmi_predictions(self):
        arr = Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1)
        results = Assets.make_BMI_predictions(arr)
        self.assertEqual(int(results[0]), 24)

    # Test if  age and gender predication function works
    def test_make_age_gender_predictions(self):
        arr = Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1)
        results = Assets.make_AgeGender_predictions(arr)
        if int(results[0]) == 13 and int(results[1]) == 0:
            self.assertTrue(True)
        else:
            self.fail()

    # Test if height and weight predication function works
    def test_make_height_weight_predictions(self):
        arr = Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1)
        results = Assets.make_HeightWeight_predictions(arr)
        if int(results[0]) == 69 and int(results[1]) == 167:
            self.assertTrue(True)
        else:
            self.fail()

    # test if pounds to kilogram converter works correctly
    def test_pounds_to_kg(self):
        # 10 pounds = 4.53592 kg
        actual = float("{0:.5f}".format(Assets.poundsToKG(10)))
        expected = 4.53592
        self.assertEqual(expected, actual)

    # test if inches to centimeter converter works correctly
    def test_inches_to_cm(self):
        actual = float("{0:.1f}".format(Assets.inchesToCm(10)))
        expected = 25.4
        self.assertEqual(expected, actual)

    # Test if the BMR calculator works correctly
    def test_aghwto_bmr(self):
        arr = Assets.img2arr('./website/CropedPhotoTemp/cropedForTest.jpg', 1)
        ageGender = Assets.make_AgeGender_predictions(arr)
        heightWeight = Assets.make_HeightWeight_predictions(arr)

        height = Assets.inchesToCm(heightWeight[0])
        weight = Assets.poundsToKG(heightWeight[1])
        if ageGender[1] > 0.6:
            BMR = (10 * weight) + (6.25 * height) - (5 * int(ageGender[0])) + 5
        else:
            BMR = (10 * weight) + (6.25 * height) - (5 * int(ageGender[0])) - 161

        self.assertEqual(BMR, Assets.AGHWToBMR(ageGender[0], ageGender[1], heightWeight[0], heightWeight[1]))
