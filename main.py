from flask import Flask, request, jsonify
import ml_model
import gunicorn
import seaborn

app = Flask('app')




@app.route('/getbmi', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getbmi():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    ml_model.crop_save_image()
    arr = ml_model.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = ml_model.make_BMI_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)


@app.route('/getagegender', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getagegender():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    ml_model.crop_save_image()
    arr = ml_model.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = ml_model.make_AgeGender_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)

@app.route('/getheightweight', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getheightweight():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    ml_model.crop_save_image()
    arr = ml_model.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = ml_model.make_HeightWeight_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)

@app.route('/getheightweightagegender', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getheightweightagegender():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    ml_model.crop_save_image()
    arr = ml_model.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictionsAG = ml_model.make_AgeGender_predictions(arr)
    predictionsHW = ml_model.make_HeightWeight_predictions(arr)

    result = {
        'PIC_prediction': list(predictionsAG+predictionsHW)
    }
    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model is really use full!"


if __name__ == '__main__':
    app.run()

