from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from routes import setup_routes

app = Flask(__name__)
Bootstrap(app)
setup_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
