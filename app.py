from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Spotify_Popularity = float(request.form['Spotify_Popularity'])
    All_Time_Rank = float(request.form['All_Time_Rank'])
    Amazon_Playlist_Count = float(request.form['Amazon_Playlist_Count'])


    input_data = np.array([[Spotify_Popularity, All_Time_Rank, Amazon_Playlist_Count ]])
    predicted_score = model.predict(input_data)*1.05

    return render_template('result.html', predicted_price=round(predicted_score[0]))

if __name__ == '__main__':
    app.run(debug=True)
