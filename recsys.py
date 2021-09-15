import bz2
import sys
import pickle
import pandas as pd
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


# def artists_recommendation(query_index, m, id_artist_map, n_rec):
#     recommendations = []
#     distances, indices = m.kneighbors(id_artist_map[id_artist_map['id'] == query_index].user_artist_count.values[0],
#                                       n_neighbors=n_rec)
#     for i in range(0, len(distances.flatten())):
#         if i == 0:
#             continue
#         else:
#             recommendations.append(id_artist_map[id_artist_map['id'] == indices.flatten()[i]].artist.values[0])
#
#     return recommendations

def get_artist_object_for_json(artist_defs):
    allartists = get_artists_data()
    artist_ids, artists, followes, imageUrls = [], [], [], []
    for artist_def in artist_defs:
        artist_ids.append(artist_def)
        artists.append(allartists[allartists['ArtistId'] == artist_def].ArtistName.values[0])
        followes.append(int(allartists[allartists['ArtistId'] == artist_def].follower.values[0]))
        imageUrls.append(allartists[allartists['ArtistId'] == artist_def].ImageUrl.values[0])

    artists = pd.DataFrame({
        "ArtistId": artist_ids,
        "ArtistName": artists,
        "Follower": followes,
        "ImageUrl": imageUrls
    }, columns=['ArtistId', "ArtistName", "Follower", "ImageUrl"])

    return artists


def get_song_object_for_json(ids):
    songs, song_ids, artists, artist_ids, durations, contentTypes, playUrls, imageUrls, copyrights, labelNames = \
        [], [], [], [], [], [], [], [], [], []

    allsongs = get_song_dataset()
    allartists = get_artists_data()

    for song_def in ids:
        song = (allsongs[allsongs['ContentId'] == song_def]).ContentName.values[0]
        song_id = allsongs[allsongs['ContentId'] == song_def].ContentId.values[0]
        artist = allsongs[allsongs['ContentId'] == song_def].Artist.values[0]
        artist_id = allartists[allartists['ArtistName'] == artist].ArtistId.values[0]
        duration = allsongs[allsongs['ContentId'] == song_def].Duration.values[0]
        contentType = allsongs[allsongs['ContentId'] == song_def].Type.values[0]
        playUrl = allsongs[allsongs['ContentId'] == song_def].PlayUrl.values[0]
        imageUrl = allsongs[allsongs['ContentId'] == song_def].ImageUrl.values[0]
        copyright = allsongs[allsongs['ContentId'] == song_def].Copyright.values[0]
        labelName = allsongs[allsongs['ContentId'] == song_def].LabelName.values[0]

        songs.append(song)
        artists.append(artist)
        song_ids.append(int(song_id))
        artist_ids.append(int(artist_id))
        durations.append(int(duration))
        contentTypes.append(contentType)
        playUrls.append(playUrl)
        imageUrls.append(imageUrl)
        copyrights.append(copyright)
        labelNames.append(labelName)

    song_df = pd.DataFrame({
        "ContentId": song_ids,
        "ContentName": songs,
        "ArtistId": artist_ids,
        "ArtistName": artists,
        "Duration": durations,
        "ContentType": contentTypes,
        "PlayUrl": playUrls,
        "ImageUrl": imageUrls,
        "Copyright": copyrights,
        "Label": labelNames

    }, columns=['ContentId', 'ContentName', 'ArtistId', 'ArtistName', "Duration", "ContentType", "PlayUrl",
                "ImageUrl", "Copyright", "Label"])

    return song_df


def artists_recommendation(artist_id, model_knn_artist, artist_pivot, n, allartists):
    artist_ids, artists, followes, imageUrls = [], [], [], []

    try:
        distances, indices = model_knn_artist.kneighbors(artist_pivot[artist_pivot.index.get_level_values('ArtistId') ==
                                                                      artist_id].
                                                         values.reshape(1, -1), n_neighbors=n + 1)
        for i in range(0, len(distances.flatten())):
            if i == 0:
                continue
            else:
                artist_def = (artist_pivot.index[indices.flatten()[i]])
                artist_ids.append(int(artist_def))
                artists.append(allartists[allartists['ArtistId'] == artist_def].ArtistName.values[0])
                followes.append(int(allartists[allartists['ArtistId'] == artist_def].follower.values[0]))
                imageUrls.append(allartists[allartists['ArtistId'] == artist_def].ImageUrl.values[0])

    except ValueError:
        pass

    return artist_ids, artists, followes, imageUrls


def song_recommendation(song_id, model_knn, pivot_table, n_rec, allsongs, allartists):
    songs, song_ids, artists, artist_ids, durations, contentTypes, playUrls, imageUrls, copyrights, labelNames = \
        [], [], [], [], [], [], [], [], [], []

    try:
        distances, indices = model_knn.kneighbors(
            pivot_table[pivot_table.index.get_level_values('ContentId') == song_id].
                values.reshape(1, -1), n_neighbors=n_rec + 1)

        for i in range(0, len(distances.flatten())):
            if i == 0:
                continue
            else:
                song_def = (pivot_table.index[indices.flatten()[i]])
                song = (allsongs[allsongs['ContentId'] == song_def]).ContentName.values[0]
                song_id = allsongs[allsongs['ContentId'] == song_def].ContentId.values[0]
                artist = allsongs[allsongs['ContentId'] == song_def].Artist.values[0]
                artist_id = allartists[allartists['ArtistName'] == artist].ArtistId.values[0]
                duration = allsongs[allsongs['ContentId'] == song_def].Duration.values[0]
                contentType = allsongs[allsongs['ContentId'] == song_def].Type.values[0]
                playUrl = allsongs[allsongs['ContentId'] == song_def].PlayUrl.values[0]
                imageUrl = allsongs[allsongs['ContentId'] == song_def].ImageUrl.values[0]
                copyright = allsongs[allsongs['ContentId'] == song_def].Copyright.values[0]
                labelName = allsongs[allsongs['ContentId'] == song_def].LabelName.values[0]

                songs.append(song)
                artists.append(artist)
                song_ids.append(int(song_id))
                artist_ids.append(int(artist_id))
                durations.append(int(duration))
                contentTypes.append(contentType)
                playUrls.append(playUrl)
                imageUrls.append(imageUrl)
                copyrights.append(copyright)
                labelNames.append(labelName)


    except ValueError:
        songs, song_ids, artists, artist_ids, durations, contentTypes, playUrls, imageUrls, copyrights, labelNames = \
            [], [], [], [], [], [], [], [], [], []

    return songs, song_ids, artists, artist_ids, durations, contentTypes, playUrls, imageUrls, copyrights, labelNames


def get_artists_songs_for_onboarding_users(artist_chosen, songs_chosen, pivot_songs, pivot_artists,
                                           model_knn, model_knn_artist, nrecartist, nrecsongs):
    allsongs = get_song_dataset()
    allartists = get_artists_data()

    onboarding_songids, onboarding_songs, songartist, songartistids, durations, \
    contentTypes, playUrls, images, copyrights, labels \
        = \
        [], [], [], [], [], [], [], [], [], []
    for song_id in songs_chosen:

        s, sid, a, aid, dur, cont, playurl, img, copyr, label \
            = song_recommendation(song_id, model_knn, pivot_songs, nrecsongs, allsongs, allartists)

        for osid in sid:
            onboarding_songids.append(osid)
        for song in s:
            onboarding_songs.append(song)
        for d in dur:
            durations.append(d)
        for c in cont:
            contentTypes.append(c)
        for p in playurl:
            playUrls.append(p)
        for ai in aid:
            songartistids.append(ai)
        for sa in a:
            songartist.append(sa)
        for im in img:
            images.append(im)
        for cop in copyr:
            copyrights.append(cop)
        for lab in label:
            labels.append(lab)

    song_df = pd.DataFrame({
        "ContentId": onboarding_songids,
        "ContentName": onboarding_songs,
        "ArtistId": songartistids,
        "ArtistName": songartist,
        "Duration": durations,
        "ContentType": contentTypes,
        "PlayUrl": playUrls,
        "ImageUrl": images,
        "Copyright": copyrights,
        "Label": labels

    }, columns=['ContentId', 'ContentName', 'ArtistId', 'ArtistName', "Duration", "ContentType", "PlayUrl",
                "ImageUrl", "Copyright", "Label"])

    onboarding_artists, obfollows, obartistImageUrls, onboarding_artistids = [], [], [], []

    for artist_id in artist_chosen:
        aids, artists, follows, artistImageUrls = artists_recommendation(artist_id, model_knn_artist, pivot_artists,
                                                                         nrecartist, allartists)
        for oaid in aids:
            onboarding_artistids.append(oaid)
        for oa in artists:
            onboarding_artists.append(oa)
        for follow in follows:
            obfollows.append(follow)
        for im in artistImageUrls:
            obartistImageUrls.append(im)

    artist_df = pd.DataFrame({
        "ArtistId": onboarding_artistids,
        "ArtistName": onboarding_artists,
        "Follows": obfollows,
        "ImageUrl": obartistImageUrls
    }, columns=['ArtistId', 'ArtistName', "Follows", "ImageUrl"])

    return song_df, artist_df


def get_pivot_songs():
    with open('pivot_table_songs.csv', 'rb') as file:
        # df = pickle.Unpickler(file).load()
        df = joblib.load('pivot_table_songs.csv')
        file.close()

    return df


import joblib


def get_pivot_artists():
    with open('pivot_table_artists.csv', 'rb') as file:
        # df = pickle.Unpickler(file).load()
        df = joblib.load('pivot_table_artists.csv')
        file.close()

    return df


def get_srm():
    song_rs_model = 'song_rs_model.pkl'

    with open(song_rs_model, 'rb') as file:
        srm = pickle.load(file)

    return srm


def get_arm():
    artist_rs_model = 'artist_rs_model.pkl'

    with open(artist_rs_model, 'rb') as file:
        arm = pickle.load(file)

    return arm


def get_all_songs():
    songdata = pd.read_csv('all_songs.csv')
    return songdata[['ContentId', 'ContentName']]


def get_all_artists():
    artistdata = pd.read_csv('all_artists.csv')
    return artistdata[['ArtistId', 'ArtistName']]


def get_song_dataset():
    songdata = pd.read_csv('all_songs.csv')
    return songdata


def get_artists_data():
    artistdata = pd.read_csv('all_artists.csv')
    return artistdata


def get_song_recommendation_selective(song, top):
    song_pivot = get_pivot_songs()
    srm = get_srm()

    nrec = top

    all_songs = get_song_dataset()
    all_artists = get_artists_data()

    try:
        # query_index = np.where(song_pivot.index.get_level_values('ContentName') == song)[0][0]

        s, sid, a, aid, dur, cont, playurl, img, copyr, label \
            = song_recommendation(song, srm, song_pivot, nrec, all_songs, all_artists)
        song_df = pd.DataFrame({
            "ContentId": sid,
            "ContentName": s,
            "ArtistId": aid,
            "ArtistName": a,
            "Duration": dur,
            "ContentType": cont,
            "PlayUrl": playurl,
            "ImageUrl": img,
            "Copyright": copyr,
            "Label": label

        }, columns=['ContentId', 'ContentName', 'ArtistId', 'ArtistName', "Duration", "ContentType", "PlayUrl",
                    "ImageUrl", "Copyright", "Label"])


    except:
        return None

    return song_df


def get_song_recommendation(req):
    song_pivot = get_pivot_songs()
    srm = get_srm()

    song = req['song_id']
    nrec = req['n_rec']

    all_songs = get_song_dataset()
    all_artists = get_artists_data()

    try:
        # query_index = np.where(song_pivot.index.get_level_values('ContentName') == song)[0][0]

        s, sid, a, aid, dur, cont, playurl, img, copyr, label \
            = song_recommendation(song, srm, song_pivot, nrec, all_songs, all_artists)
        song_df = pd.DataFrame({
            "ContentId": sid,
            "ContentName": s,
            "ArtistId": aid,
            "ArtistName": a,
            "Duration": dur,
            "ContentType": cont,
            "PlayUrl": playurl,
            "ImageUrl": img,
            "Copyright": copyr,
            "Label": label

        }, columns=['ContentId', 'ContentName', 'ArtistId', 'ArtistName', "Duration", "ContentType", "PlayUrl",
                    "ImageUrl", "Copyright", "Label"])


    except:
        return None

    return song_df


def get_artist_recommendation_selective(artist, top):
    allartists = get_artists_data()
    artist_pivot = get_pivot_artists()
    arm = get_arm()

    try:
        aid, a, follows, imageUrls = artists_recommendation(artist, arm, artist_pivot, top, allartists)

        artists = pd.DataFrame({
            "ArtistId": aid,
            "ArtistName": a,
            "Follower": follows,
            "ImageUrl": imageUrls
        }, columns=['ArtistId', "ArtistName", "Follower", "ImageUrl"])

        return artists

    except:
        return None


def get_artists_recommendation(req):
    artist_pivot = get_pivot_artists()
    arm = get_arm()

    artist_id = req['artist_id']
    nrec = req['n_rec']

    allartists = get_artists_data()

    try:
        aid, a, follows, imageUrls = artists_recommendation(artist_id, arm, artist_pivot, nrec, allartists)

        artists = pd.DataFrame({
            "ArtistId": aid,
            "ArtistName": a,
            "Follower": follows,
            "ImageUrl": imageUrls
        }, columns=['ArtistId', "ArtistName", "Follower", "ImageUrl"])

        return artists

    except:
        return None


def get_songuser_data():
    return pd.read_csv("songuser.csv")


def get_songuser_with_time():
    return pd.read_csv("songusertime.csv")


def get_songs_artists_onboarding_user(req):
    list_artists = req['artist_ids']
    list_songs = req['song_ids']
    n_recartists = req['n_rec_artists']
    n_rec_songs = req['n_rec_songs']
    song_pivot = get_pivot_songs()
    artist_pivot = get_pivot_artists()
    srm = get_srm()
    arm = get_arm()

    return get_artists_songs_for_onboarding_users(list_artists, list_songs, song_pivot, artist_pivot, srm, arm,
                                                  n_recartists, n_rec_songs)


def get_popular_song_recommendations(req):
    n = req['n_rec']

    try:
        songuser_v2 = get_songuser_data()
        allsongs = get_all_songs()

        popsongs = pd.DataFrame(
            songuser_v2[['ContentId', 'ArtistId', 'SongStreamCount']].groupby('ContentId', as_index=False)[
                'SongStreamCount'].sum())

        popsongs['Rank'] = popsongs['SongStreamCount'].rank(ascending=0, method='first')

        popular_songs = popsongs.sort_values(['SongStreamCount', 'ContentId'], ascending=[0, 1])

        popular_song_recommendations = pd.merge(popular_songs, allsongs, on='ContentId', how='left')[
            ['ContentId', 'ContentName', 'SongStreamCount', 'Rank']].head(n)

        song_ids = popular_song_recommendations.ContentId.unique().tolist()

        print(song_ids)

        song_df = get_song_object_for_json(song_ids)

        print("bhung")

        return song_df

    except ValueError:
        return None


def get_popular_artists_recommendation(req):
    n = req['n_rec']

    try:
        songuser_v2 = get_songuser_data()


        popartist = pd.DataFrame(
            songuser_v2[['ArtistId', 'ArtistName', 'SongStreamCount']].groupby(['ArtistId', 'ArtistName'],
                                                                               as_index=False)['SongStreamCount'].sum())
        popartist['Rank'] = popartist['SongStreamCount'].rank(ascending=0, method='first')


        popular_artists = popartist.sort_values(['SongStreamCount', 'ArtistId'], ascending=[0, 1]).head(n)

        artist_ids = popular_artists.ArtistId.unique().tolist()
        artist_df = get_artist_object_for_json(artist_ids)

        return artist_df
    except:
        return None


def get_discover_new_artist(req):
    user = req['user_id']
    n = req['n_rec']

    try:
        songuser_v2 = get_songuser_data()

        fav_artists_id = songuser_v2[songuser_v2['Misisdn'] == user].sort_values(['SongStreamCount'],
                                                                                 ascending=False).head(
            5).ArtistId.tolist()
        all_rec = []
        all_artists = songuser_v2[songuser_v2['Misisdn'] == user].sort_values(['SongStreamCount'],
                                                                              ascending=False).ArtistId.tolist()
        for artist in fav_artists_id:
            for rec in get_artist_recommendation_selective(artist, 5).ArtistId.tolist():
                all_rec.append(rec)

        artists_tobe_discovered = list(set(all_rec) - set(all_artists))

        artist_df = get_artist_object_for_json(artists_tobe_discovered)

        return artist_df.head(n)

    except:
        return None


def get_weekly_playlist(req):
    songusertime = get_songuser_with_time()

    user = req['user_id']
    top = req['n_rec']

    try:
        recent_songs = songusertime[songusertime['Misisdn'] == user].sort_values(['Date'],
                                                                                 ascending=False).ContentId.unique().tolist()[
                       :15]
        playlist = []

        for song in recent_songs:
            for rec in get_song_recommendation_selective(song, 1).ContentId.tolist():
                playlist.append(rec)

        song_df = get_song_object_for_json(playlist)

        return song_df

    except:
        return None