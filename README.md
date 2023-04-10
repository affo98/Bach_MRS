# Bachelor in Automated Playlist Continuation with Data From Spotify RecSys Challenge 2018

This project aims to evaluate different RecSys algorithms on the spectrum of Serendipity, Novelty and Diversity in their respective recommendations on the task of Automated Playlist Continuation (APC).

The two folders of MPDS respectively spot_mpd (training) and spot_mpdc(validation) hold the data for all playlists in the structure of a split json files into chunks of maximum playlist capacity per json is set to a 1000. 
Each playlist entry in the json files is listed as:
  
        {
            "name": "musical",
            "collaborative": "false",
            "pid": 5,
            "modified_at": 1493424000,
            "num_albums": 7,
            "num_tracks": 12,
            "num_followers": 1,
            "num_edits": 2,
            "duration_ms": 2657366,
            "num_artists": 6,
            "tracks": [
                {
                    "pos": 0,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:7vqa3sDmtEaVJ2gcvxtRID",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Finalement",
                    "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                    "duration_ms": 166264,
                    "album_name": "Dancing Chords and Fireflies"
                },
                {
                    "pos": 1,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:23EOmJivOZ88WJPUbIPjh6",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Betty",
                    "album_uri": "spotify:album:3lUSlvjUoHNA8IkNTqURqd",
                    "duration_ms": 235534,
                    "album_name": "Endless Smile"
                },
                {
                    "pos": 2,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:1vaffTCJxkyqeJY7zF9a55",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Some Beat in My Head",
                    "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                    "duration_ms": 268050,
                    "album_name": "Dancing Chords and Fireflies"
                },
                // 8 tracks omitted
                {
                    "pos": 11,
                    "artist_name": "Mo' Horizons",
                    "track_uri": "spotify:track:7iwx00eBzeSSSy6xfESyWN",
                    "artist_uri": "spotify:artist:3tuX54dqgS8LsGUvNzgrpP",
                    "track_name": "Fever 99\u00b0",
                    "album_uri": "spotify:album:2Fg1t2tyOSGWkVYHlFfXVf",
                    "duration_ms": 364320,
                    "album_name": "Come Touch The Sun"
                }
            ],

        }
\n

To load the data and create the user x item matrix run the DataProcessing.py that iterates through each playlist and stores the implicit feedback in a binary sci-py sparse matrix, where each nonzero entry is deemed as a track the user likes since it is a part of the playlist. Some users have put tracks on their playlists multiple times, but for the sake of comparing different methods of novelty, serendipity and diversity, a weighting scheme of the tracks occuring more than once will be ignored, to avoid complicating the preprocessing with a weightin scheme. To validate the user item matrix generated and stored in "UserItemMatrix.npz", the script DataLoadEval.py takes a random playlist from the first 1000 playlists and evaluates the lookuptable "SongIdToIndex.py", by going through the playlist track URIs and comparing them to the {key (uri) : value (column index in matrix)} and match their URI's. It returns True and if the Indexes match and is run as part of the "DataProcessing.py"


