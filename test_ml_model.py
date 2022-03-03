from unittest import TestCase
from ml_model import Asset

Assets = Asset()


class TestAsset(TestCase):
    # def test_detect_face(self):
    #     self.fail()
    #
    # def test_save_image(self):
    #     self.fail()
    #
    # def test_crop_save_image(self):
    #     self.fail()
    #
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
    # def test_inches_to_cm(self):
    #     self.fail()

    # def test_aghwto_bmr(self):
    #     self.fail()

