from flask import Flask, render_template, url_for
from flask_bootstrap import Bootstrap
from routes import setup_routes

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management
Bootstrap(app)
setup_routes(app)

@app.context_processor
def inject_navigation():
    return dict(navigation=[
        {'name': 'Home', 'url': url_for('index')},
        {'name': 'Select Algorithm', 'url': url_for('select_algorithm')}
    ])

if __name__ == "__main__":
    app.run(debug=True)
