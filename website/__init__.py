from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'TeamFeugo'

    from .views import views
    from .ml import ml

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(ml, url_prefix='/')
    return app
