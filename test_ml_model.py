from unittest import TestCase
from ml_model import Asset
import os
Assets = Asset('./NotCropedPhoto/temp.jpg', './CropedPhotoTemp/cropedForTest.jpg', "./models/BMI_f16.tflite",
               "./models/AgeGender_fp16.tflite", "./models/height_weight_models.tflite")


class TestAsset(TestCase):
    def test_detect_face(self):
        try:
            Assets.detect_face()
            self.assertTrue(True)
        except:
            self.fail()


    def test_crop_save_image(self):
        if os.path.exists(Assets.test_processed_dir):
            os.remove(Assets.test_processed_dir)

        Assets.crop_save_image()

        if os.path.exists(Assets.test_processed_dir):
            self.assertTrue(True)
        else:
            self.fail()


    # def test_img2arr(self):
    #     self.fail()
    #
    # def test_load_model_bmi(self):
    #     self.fail()
    #
    # def test_load_model_age_gender(self):
    #     self.fail()
    #
    # def test_load_model_height_weight(self):
    #     self.fail()
    #
    # def test_make_bmi_predictions(self):
    #     self.fail()
    #
    # def test_make_age_gender_predictions(self):
    #     self.fail()
    #
    # def test_make_age_gender_predictions(self):
    #     self.fail()
    #
    # def test_make_height_weight_predictions(self):
    #     self.fail()
    #
    def test_pounds_to_kg(self):
        # 10 pounds = 4.53592 kg
        actual = float("{0:.5f}".format(Assets.poundsToKG(10)))
        expected = 4.53592
        self.assertEqual(expected, actual)

    #
    def test_inches_to_cm(self):
        actual = float("{0:.1f}".format(Assets.inchesToCm(10)))
        expected = 25.4
        self.assertEqual(expected, actual)

    # def test_aghwto_bmr(self):
    #     self.fail()
