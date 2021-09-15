from flask import Flask, request, json, make_response, Response
from flask import jsonify
import recsys
import pandas as pd

app = Flask(__name__)


@app.route('/shadhinmre_v1/songs', methods=['GET'])
def get_song_recommendations():
    req = request.get_json()
    song_df = recsys.get_song_recommendation(req)

    if song_df is not None:
        response = {
            "recommended_songs":song_df.to_dict(orient='records')
        }
    else:
        response = {
            "recommended_songs": []
        }

    return make_response(jsonify(response), 200)


@app.route('/shadhinmre_v1/artists', methods=['GET'])
def get_artists_recommendations():
    req = request.get_json()
    artist_df = recsys.get_artists_recommendation(req)

    # if artist_ids is None:
    #     artist_ids, artists = [], []

    if artist_df is not None:
        response = {
            "recommended_artists": artist_df.to_dict(orient='records')
        }
    else:
        response = {
            "recommended_artists":[]
        }

    return make_response(jsonify(response), 200)


@app.route('/shadhinmre_v1/onboard', methods=['GET'])
def get_onboard_songs_and_artists():
    req = request.get_json()
    song_df, artist_df = recsys.get_songs_artists_onboarding_user(req)


    response = {
        "recommended_onboarding_songs": song_df.to_dict(orient='records'),
        "recommended_onboarding_artists": artist_df.to_dict(orient='records')
    }
    return make_response(jsonify(response), 200)


@app.route('/shadhinmre_v1/allsongs', methods=['GET'])
def get_all_songs():
    allsongs = recsys.get_all_songs()
    response = {
        'all_songs': (allsongs.to_dict(orient='records'))
    }
    return make_response(jsonify(response), 200)


@app.route('/shadhinmre_v1/allartists', methods=['GET'])
def get_all_artists():
    allartists = recsys.get_all_artists()

    response = {
        'all_artists': (allartists.to_dict(orient='records'))
    }
    return make_response(jsonify(response), 200)

@app.route('/shadhinmre_v1/popularsongrec', methods=['GET'])
def get_popular_song_rec():
    req = request.get_json()
    songs = recsys.get_popular_song_recommendations(req)

    response = {
        'popular_song_recommendations': (songs.to_dict(orient='records'))
    }
    return make_response(jsonify(response), 200)

@app.route('/shadhinmre_v1/popularartistsrec', methods=['GET'])
def get_popular_artist_rec():
    req = request.get_json()

    artists = recsys.get_popular_artists_recommendation(req)

    response = {
        'popular_artist_recommendations': (artists.to_dict(orient='records'))
    }

    return make_response(jsonify(response), 200)


@app.route('/shadhinmre_v1/discoverartists', methods=['GET'])
def get_new_discovered_artists():
    req = request.get_json()

    artists = recsys.get_discover_new_artist(req)

    response = {
        'discovered_artists': (artists.to_dict(orient='records'))
    }

    return make_response(jsonify(response), 200)

@app.route('/shadhinmre_v1/weeklyplaylist', methods=['GET'])
def get_weekly_playlists():
    req = request.get_json()

    songs = recsys.get_weekly_playlist(req)

    response = {
        'weekly_recommended_playlists': (songs.to_dict(orient='records'))
    }

    return make_response(jsonify(response), 200)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
