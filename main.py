import json
from flask import Flask, request, jsonify
from ml_model import Asset


app = Flask('app')
Assets = Asset('./NotCropedPhoto/temp.jpg', './CropedPhotoTemp/croped.jpg', "./models/BMI_f16.tflite",
               "./models/AgeGender_fp16.tflite", "./models/height_weight_models.tflite")


@app.route('/', methods=['GET'])
def homePage():
    return "This is home page"


@app.route('/faceDetect', methods=['POST'])
def faceDetect():
    # getting image data from post request
    file = request.files['image']
    # saving image
    file.save("NotCropedPhoto/temp.jpg")
    # ckeck if face detectable
    faceDetectAble = True
    try:
        Assets.crop_save_image()
    except:
        faceDetectAble = False
    # create result
    result = {
        'PIC_prediction': str(faceDetectAble)
    }
    return jsonify(result)


@app.route('/getbmi', methods=['POST'])  # get BMI from image
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
           methods=['POST'])
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
           methods=['POST'])
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


@app.route('/addData', methods=['POST'])
def post_users():
    data = request.data.decode('utf-8')
    jsondata = json.loads(data)
    # file = request.file['username']
    # _username = 'TharinduSilva'
    # _BMI = 24.1
    # _BMR = 1775
    _username = jsondata['username']
    _BMI = float(jsondata['BMI'])
    _BMR = int(jsondata['BMR'])

    sql = f"INSERT INTO userdata (username, BMI, BMR) VALUES ('{_username}', '{_BMI}','{_BMR}');"

    Assets.execute(sql, True)
    return "I hope it worked"


@app.route('/getData', methods=['POST'])
def getData():
    data = request.data.decode('utf-8')
    jsondata = json.loads(data)

    _username = jsondata['username']
    # sql = f"SELECT BMI, BMR FROM userdata WHERE username='{_username}';"
    sql = f"SELECT BMI,BMR FROM userdata where username='{_username}';"
    IHopeItWorked = Assets.execute(sql, False)
    print(IHopeItWorked)
    BMI = []
    BMR = []
    for i in IHopeItWorked:
        BMI.append(i[0])
        BMR.append(i[1])

    result = {
        'BMI_data': BMI, 'BMR_data': BMR
    }
    return result


@app.route('/PostTest', methods=['POST'])
def PostTest():
    data = request.data.decode('utf-8')
    print(type(data))
    jsondata = json.loads(data)
    print(jsondata['username'])
    print(jsondata['BMI'])
    print(jsondata['BMR'])
    return data


if __name__ == '__main__':
    app.run()
