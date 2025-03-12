from flask import Flask, request, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from routes import setup_routes
from StockData import StockData
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
Bootstrap(app)

stock_data = StockData(tickers='AAPL')  # Initialize with a default ticker

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    stock_data.set_ticker_and_dates(ticker, start_date, end_date)
    return redirect(url_for('select_algorithm'))

@app.route('/select_algorithm')
def select_algorithm():
    return render_template('select_algorithm.html')

@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    algorithm = request.form['algorithm']
    if algorithm == 'regression':
        # Example regression algorithm
        df = stock_data.get_data()
        df['min_1_close'] = df['close'].shift(1)
        df.dropna(inplace=True)
        X = df[['min_1_close']]
        y = df['close']
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        df['predicted_close'] = model.predict(X)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['close'], label='Actual Close')
        plt.plot(df.index, df['predicted_close'], label='Predicted Close')
        plt.legend()
        plt.title(f'{ticker} Close Price Prediction')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('result.html', plot_url=plot_url)
    return "Algorithm not implemented", 400

if __name__ == "__main__":
    app.run(debug=True)
