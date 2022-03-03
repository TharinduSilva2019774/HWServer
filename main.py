from flask import Flask, request, jsonify
from ml_model import Asset
import gunicorn
import seaborn

app = Flask('app')
Assets = Asset()


@app.route('/faceDetect', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def faceDetect():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    faceDetectAble = True
    try:
        Assets.crop_save_image()
    except:
        faceDetectAble = False

    result = {
        'PIC_prediction': str(faceDetectAble)
    }
    return jsonify(result)


@app.route('/getbmi', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getbmi():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = Assets.make_BMI_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)


@app.route('/getagegender',
           methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getagegender():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = Assets.make_AgeGender_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)


@app.route('/getheightweight',
           methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getheightweight():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = Assets.make_HeightWeight_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)


@app.route('/getheightweightagegender',
           methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getheightweightagegender():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictionsAG = Assets.make_AgeGender_predictions(arr)
    predictionsHW = Assets.make_HeightWeight_predictions(arr)

    result = {
        'PIC_prediction': list(predictionsAG + predictionsHW)
    }
    return jsonify(result)


@app.route('/getbmr', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getBMR():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictionsAG = Assets.make_AgeGender_predictions(arr)
    predictionsHW = Assets.make_HeightWeight_predictions(arr)
    BMR = Assets.AGHWToBMR(predictionsAG[0], predictionsAG[1], predictionsHW[0], predictionsHW[1])

    result = {
        'PIC_prediction': int(BMR)
    }
    return jsonify(result)


@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model is really use full! Yay!!!!!!!!!"


if __name__ == '__main__':
    app.run()
