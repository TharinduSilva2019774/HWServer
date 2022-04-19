import json
from flask import Flask, request, jsonify, render_template
from ml_model import Asset
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from flask import Blueprint, render_template, request, flash, redirect, url_for

from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user

from flask_login import UserMixin
from sqlalchemy.sql import func

db = SQLAlchemy()
DB_NAME = "database.db"

app = Flask('app')
Assets = Asset('./NotCropedPhoto/temp.jpg', './CropedPhotoTemp/croped.jpg', "./models/BMI_f16.tflite",
               "./models/AgeGender_fp16.tflite", "./models/height_weight_models.tflite")

app.config['SECRET_KEY'] = 'TeamFeugo'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
db.init_app(app)


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')

create_database(app)


login_manager = LoginManager()
login_manager.login_view = 'app.login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    notes = db.relationship('Note')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('home'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account created!', category='success')
            return redirect(url_for('home'))

    return render_template("sign_up.html", user=current_user)


@app.route('/home', methods=['GET'])
def home():
    return render_template("home.html")


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
    _username = jsondata['username']
    _BMI = float(jsondata['BMI'])
    _BMR = int(jsondata['BMR'])
    sql = f"INSERT INTO userdata (username, BMI, BMR) VALUES ('{_username}', '{_BMI}','{_BMR}');"
    Assets.execute(sql, True)
    return "worked"


@app.route('/getData', methods=['POST'])
def getData():
    data = request.data.decode('utf-8')
    jsondata = json.loads(data)

    _username = jsondata['username']
    sql = f"SELECT BMI,BMR FROM userdata where username='{_username}';"
    result = Assets.execute(sql, False)
    print(result)
    BMI = []
    BMR = []
    for i in result:
        BMI.append(i[0])
        BMR.append(i[1])

    result = {
        'BMI_data': BMI, 'BMR_data': BMR
    }
    return result


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./NotCropedPhoto/temp.jpg')
        Assets.crop_save_image()
        arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

        BMI = Assets.make_BMI_predictions(arr)
        predictionsAG = Assets.make_AgeGender_predictions(arr)
        predictionsHW = Assets.make_HeightWeight_predictions(arr)
        BMR = Assets.AGHWToBMR(predictionsAG[0], predictionsAG[1], predictionsHW[0], predictionsHW[1])
        print(BMI)
        print(BMR)
        return render_template("result.html", BMI=round(BMI[0], 2), BMR=int(BMR))


if __name__ == '__main__':
    app.run()
