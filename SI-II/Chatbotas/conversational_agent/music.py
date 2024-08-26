import requests
import random
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import cred
import contextlib
import warnings

#Supress Warnings
warnings.filterwarnings("ignore")

class MusicInfo:
    def __init__(self) -> None:

        self.patterns      = ["music", "artist", "song", "listen", "track", "musician"]
        self.responses     = ["I don't have the answer that you looking for.\n\t", "You can google about it ;)\n\t", "I'm sorry, but I can't help you there.\n\t"]

        # LastFM API info
        self.base_url      = "https://ws.audioscrobbler.com/2.0/"
        self.api_key       = "1290eb60796a28ac933fad4254cd3f49"
        
        
    def get_patterns(self):
        return self.patterns
    

    def get_responses(self):
        return self.responses
    
    
    def get_artist_info(self, artist_name):
        if ' ' in artist_name: artist_name = artist_name.replace(" ", "%20")

        url = self.base_url + f"?method=artist.getinfo&artist={artist_name}&api_key={self.api_key}&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            document = response.json()

            if "error" not in document.keys():
                info = document["artist"]
                
                name      = info["name"]
                listeners = info["stats"]["listeners"]
                tag       = info["tags"]["tag"][0]["name"] if info["tags"]["tag"] else None
                summary   = info["bio"]["summary"]
                
                return f"This is what I know:\n - name: {name}\n - tags: {tag}\n - listeners: {listeners}\n - summary: {summary}"
            
        return "I'm sorry, I couldn't fetch the artist information."
        
        

    def get_track_info(self, track_name):

        if ' ' in track_name: track_name = track_name.replace(" ", "%20")

        url = self.base_url + f"?method=track.search&track={track_name}&api_key={self.api_key}&limit=1&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            document = response.json()

            if "error" not in document.keys():
                info = document["results"]["trackmatches"]["track"][0]

                name   = info["name"]
                artist = info["artist"]
                url    = info["url"]

                return f"This is what I know:\n - name: {name}\n - artist: {artist}\n - url: {url}"

        return "I'm sorry, I couldn't fetch the track information."
        

    def get_tracks_recommendations(self):

        url = self.base_url + f"?method=chart.gettoptracks&api_key={self.api_key}&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            document = response.json()
            if "error" not in document.keys():
                suggestions = {}
                while len(suggestions) < 3:
                    index = random.randint(0, 49)
                    if index not in suggestions.keys(): 
                        info               = document["tracks"]["track"][index]
                        name               = info["name"]
                        artist             = info["artist"]["name"]
                        suggestions[index] = (name, artist)
                
                s1, s2, s3 = suggestions.values()

                return f"My suggestions to you are:\n - {s1[0]} by {s1[1]}\n - {s2[0]} by {s2[1]}\n - {s3[0]} by {s3[1]}"

        return "I'm sorry, at the moment I can't make you any suggestions :("

    def get_tracks_by_artist(self, artist):
        
        if ' ' in artist: artist = artist.replace(" ", "%20")

        url = self.base_url + f"?method=artist.gettoptracks&artist={artist}&api_key={self.api_key}&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            document = response.json()
            if "error" not in document.keys():
                tracks = []

                num = 3
                num_of_tracks = len(document["toptracks"]["track"])
                if num_of_tracks < num:
                    num = num_of_tracks

                for _ in range(num):
                    while True:
                        index = random.randint(0, num_of_tracks - 1)
                        info  = document["toptracks"]["track"][index]
                        track = info["name"]
                        
                        if track not in tracks: 
                            artist = info["artist"]["name"]
                            tracks.append(track)
                            break

                if len(tracks) == 1:
                    return f"Here you have some tracks of {artist}:\n - {tracks[0]}\n "

                elif len(tracks) == 2:
                    return f"Here you have some tracks of {artist}:\n - {tracks[0]}\n - {tracks[1]}\n "
    
                return f"Here you have some tracks of {artist}:\n - {tracks[0]}\n - {tracks[1]}\n - {tracks[2]}"


    def get_artist_song(self, track_name, track_artist):
        if ' ' in track_name: track_name = track_name.replace(" ", "%20")
        if ' ' in track_artist: track_artist = track_artist.replace(" ", "%20")
        
        url = self.base_url+f"?method=track.getInfo&api_key={self.api_key}&artist={track_artist}&track={track_name}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            document = response.json()

            if "error" not in document.keys():

                info         = document["track"]
                name         = info["name"]
                artist       = info["artist"]["name"]
                album        = info["album"]["title"]            if "album" in info.keys() else None
                genre        = info["toptags"]["tag"][0]["name"] if info["toptags"]["tag"] else None
                plays        = info["playcount"]                 if "playcount" in info.keys() else None
                listeners    = info["listeners"]                 if "listeners" in info.keys() else None
                release_date = info["wiki"]["published"]         if "wiki" in info.keys() else None
                summary      = info["wiki"]["summary"]           if "wiki" in info.keys() else None
                
                return f"This is what I know:\n - name: {name}\n - artist: {artist}\n - album: {album}\n - genre: {genre}\n - plays: {plays}\n - listeners: {listeners}\n - release date: {release_date}\n - summary: {summary}"

        return "I'm sorry, I couldn't fetch the track."
        



class Spotify:

    def __init__(self):
        self.client_id = cred.client_ID
        self.client_secret = cred.client_SECRET
        self.redirect_uri = cred.redirect_url
        self.my_device = None
        
        self.scope = "user-read-playback-state,user-modify-playback-state,user-read-currently-playing,user-read-email,user-read-private,user-library-modify,user-library-read,user-read-playback-position,user-top-read,user-read-recently-played,user-follow-modify,user-follow-read,playlist-read-private,playlist-read-collaborative,playlist-modify-private,playlist-modify-public,app-remote-control,streaming"

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cred.client_ID, client_secret= cred.client_SECRET, redirect_uri=cred.redirect_url, scope=self.scope))
        
        self.search_for_devices()
        

    def find(self, track_name=None, artist=None):
        
        try:
            if track_name:
                results = self.sp.search(q=track_name, type='track')
            else:
                results = self.sp.search(q=artist, type='track')
        except:
            return None
        
        # meter lista ids e sauce
        songs_list={}
        artist_list={}
        self.arts =""
        c=1
        nc=1
        items = results['tracks']['items']
        if artist is None:
            for item in items:
                
                if str(track_name).lower() == str(item['name']).lower():
                    songs_list[c-1]=[item['id'],item['artists'][0]['name'], item['name']]

                    c+=1    
        else:
            if track_name is None:
                items = results['tracks']['items']
                for item in items:
                    self.arts = item['artists'][0]['name']


                    if str(self.arts.lower()) in str(artist).lower():
                        self.track = items[0]['name']
                        track_id = items[0]['id']
                        return [track_id,self.arts]
                    
                    else: 
                        return None
            else:

                for item in items:
                    if str(track_name).lower() == str(item['name']).lower() and str(artist).lower() == str(item['artists'][0]['name']).lower():
                        songs_list[c-1]=[item['id'],item['artists'][0]['name'], item['name']]
                        c+=1
        

        if len(songs_list) == 1:
            self.arts = items[0]['artists'][0]['name']
            self.track = items[0]['name']
            track_id = items[0]['id']
            return track_id
        
        elif len(songs_list) == 0:
            return None
        
        else:
            print("[bot]: These are the songs that I found:")

            for item in songs_list:
                print("\t["+ str(nc)+  '] '+ songs_list[item][1] + ' - '  +songs_list[item][2])
                nc+=1
                
            if not artist:
                print("\t[" + str(nc) + '] My song is not on the list')
            
            while True:
                try:
                    var= int(input("\nChoose a number (0 for exit): "))
                    while True:
                        if nc<var:
                            print(f"[bot]: Please enter a number lower than {nc}")
                        else:
                            break
                    break
                except:
                    print("[bot]: Please enter a number")

            if var == 0:
                return "exit"
            
            if var != nc:
                var -= 1
                track_id   = songs_list[var][0]
                self.arts  = songs_list[var][1]
                self.track = songs_list[var][2]
                return track_id
            else:
                return None
            

    def play(self, track_id):
        try:
            self.search_for_devices()
            self.sp.start_playback(device_id=self.my_device, uris=[f"spotify:track:{track_id}"])
            return f'Playing {self.arts} - {self.track}'
        
        except:
            return f"I'm getting the error above. Please, open the spotify."
    

    def pause(self):
        try:
            self.search_for_devices()
            self.sp.pause_playback(device_id=self.my_device)

        except:
            return f"I'm getting the error above. Please, open the spotify."


    def resume(self):
        try:
            self.search_for_devices()
            self.sp.start_playback(self.my_device)

        except:
            return f"I'm getting the error above. Please, open the spotify."
    
    
    def search_for_devices(self):
        for device in self.sp.devices()['devices']:
            if device['type'] == 'Smartphone':
                self.my_device = device['id']

            else:
                self.my_device = self.sp.devices()['devices'][0]['id']



if __name__ == "__main__":
    """ for testing """
    bot = MusicInfo()