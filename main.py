from flask import Flask, request, jsonify
import ml_model
import gunicorn
import seaborn

app = Flask('app')


@app.route('/getbmi', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def test():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    ml_model.crop_save_image()

    predictions = ml_model.make_predictions()
    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)



# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=9696)


@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"


if __name__ == '__main__':
    app.run()

