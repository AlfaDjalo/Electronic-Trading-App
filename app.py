from flask import Flask, render_template, url_for, session
from flask_bootstrap import Bootstrap
from routes import setup_routes

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management
Bootstrap(app)
setup_routes(app)

# Use a flag to ensure the logic runs only once
first_request_handled = False

@app.before_request
def clear_comparisons_on_start():
    global first_request_handled
    if not first_request_handled:
        session['comparisons'] = []
        first_request_handled = True

@app.context_processor
def inject_navigation():
    return dict(navigation=[
        {'name': 'Home', 'url': url_for('index')},
        {'name': 'Select Algorithm', 'url': url_for('select_algorithm')}
    ])

if __name__ == "__main__":
    app.run(debug=True)
