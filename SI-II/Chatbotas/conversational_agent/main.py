from intents import intents 
from music import MusicInfo, Spotify
from training_songs import train_artists, train_songs, learn_artist, learn_song
import spacy
import pymongo
import json, random
from pygame import mixer
import time
from spacy.matcher import PhraseMatcher
from spacy.language import Language
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="\\[W008\\] Evaluating Doc.similarity based on empty vectors")

class ConversationalBot:
    def __init__(self) -> None:
        print("\n- initializing chat bot")
        self.bot_name = "Orochi"
        self.username = None
        
        self.music_info   = MusicInfo()
        self.spotify      = Spotify()
        print("\t* starting mongo database...")
        # initialize mongodb  ( don't forget to activate the server first: sudo systemctl start mongod )
        client = pymongo.MongoClient("mongodb://localhost:27017/")

        # creating database and collection
        db = client["chatbot"]
        self.chatbot_db = db["general"]

        # insert a document into the collection
        self.all_intents = json.loads(intents)
        self.chatbot_db.insert_one(self.all_intents)

        print("\t* starting spaCy...")
        self.nlp = spacy.load("en_core_web_md")

        # Custom sentencizer function (to avoid splitting sentences on exclamation marks)
        @Language.component("custom_sentencizer")
        def custom_sentencizer(doc):
            for i, token in enumerate(doc[:-1]):
                if token.text in ["!", "?"]:
                    doc[i+1].is_sent_start = False
            return doc

        # Add the custom sentencizer to the pipeline
        self.nlp.add_pipe("custom_sentencizer", before="parser")

        self.min_similarity = 0.75
        self.learning_mode_min_similarity = 0.85

        self.learning_mode = False

        stopw_file = open("stopw.txt", "r")
        self.stop_words = list([x.strip().lower() for x in stopw_file.readlines()])
        stopw_file.close()


    def run(self):
        print("- chat bot is ready!\n")
        last_user_input = None
        intent = None

        print(f"[bot]: Hi! My name is {self.bot_name} and I am a chat bot about music. What is your name?\n")        
        while True:
            try:
                user_input = input(">> ")

                if user_input:
                    statement = self.nlp( (user_input.lower()) )

                    response = ""
                    for sentence in statement.sents:
                        intent, similarity, learned_response_pos = self.get_intent(sentence)
                        
                        if self.learning_mode and similarity < self.learning_mode_min_similarity:
                            # if the similarity is lower than the learning mode threshold, the chat bot will store the knowledge
                            self.store("learn", user_input, last_user_input)
                            response += "Thanks for increasing my knowledge!\n\t"
                            self.learning_mode = False

                        # if (not learning mode) or (in learning mode and the user writes something else)
                        else:
                            
                            if similarity < self.min_similarity: 
                                # if the similarity is lower than the threshold, the chat bot will learn
                                intent = "learn"
                                learned_response_pos = None

                            # print(f"intent = {intent} | similarity = {similarity}")
                            self.similarity = similarity

                            intent = self.update(sentence, intent)
                            self.learning_mode = False
                        
                            if intent == "timer":
                                mixer.init()
                                x = input("How many seconds? ")
                                time.sleep(float(x))
                                mixer.music.load("alarm.wav")
                                mixer.music.play()

                            elif intent != "learn": 
                                response += self.get_response(intent) + "\n\t" 

                            else:
                                # check if the chatbot have knowledge stored in the db
                                if learned_response_pos is not None:
                                    response += self.get_learned_response(learned_response_pos) + "\n\t" 

                                # if not, inform the user that we need to teach or if the theme is not music awnser other thing
                                else:
                                    #TODO: checkar se tem que ver com musica, se não tiver, mandar ir dar uma volta
                                    if any( pattern in user_input.lower() for pattern in self.music_info.get_patterns() ):
                                        # if the user input is related with music and don't know how to awswer it, learn from iteraction
                                        response += "I don't have the knowledge to answer you. What should I answer when someone asks me this?\n\t"
                                        self.learning_mode = True
                                        last_user_input = user_input 

                                    else:
                                        response += random.choice( self.music_info.get_responses() )

                    print(f"[bot]:  {response}")
                    if intent == "goodbye": break

            except (KeyboardInterrupt, EOFError, SystemExit):
                break


    def get_intent(self, statement):
        best_similarity = 0
        final_intent = None

        cursor = self.chatbot_db.find()
        document = next(cursor)
        document.pop("_id")
        document.pop("learn")

        statement = self.nlp( (self.remove_stop_words(statement.text)) )

        for intent, items in document.items():
            if "patterns" in items:
                for pattern in items["patterns"]:

                    pattern = self.nlp( (self.remove_stop_words(pattern)) )
                    similarity = pattern.similarity(statement)

                    if best_similarity < similarity:
                        best_similarity = similarity
                        final_intent = intent

        # search on learned responses, and return the position on db of the best response based on similarity
        new_similarity, learned_response_pos = self.query(statement)

        if new_similarity > best_similarity:
            best_similarity = new_similarity
            final_intent = "learn" 

        else:
            learned_response_pos = None

        if not self.username and len(statement.ents) != 0 and str(statement) == str(statement.ents[0]):
            final_intent = "greeting"
            self.username = str(statement.ents[0])
            best_similarity = 1
            
        return final_intent, best_similarity, learned_response_pos
    

    def update(self, statement, intent):
        """ Function to update some variables and the intent, if needed, depending on the knowledge of the chat bot """

        if intent == "greeting_with_name":
            # save username
            if statement.ents: 
                name = str(statement.ents[0])

            else:
                name = input("[bot]: I'm sorry, I didn't understand your username. Can you type it for me? (only your username please)\n\n>> ")

            self.username = name[0].upper() + name[1:]


        elif intent == "music_recommend":
            response = self.music_info.get_tracks_recommendations()
            self.store(intent, response)    # Update responses for the music_recommend intent


        elif intent == "music_artist":
            artist_name = train_artists(self.nlp, statement.text)
            if artist_name == None: 
                artist_name = input("[bot]: I don't have records of that artist, but I can search more deeper. Can you type the artist name again?\n\n>> ")

                new_intent, similarity, _ = self.get_intent( self.nlp(artist_name) )
                self.learning_mode = True

                if similarity < self.min_similarity:   
                    learn_artist(artist_name)
                
                else:
                    return self.update(  self.nlp(artist_name), new_intent )  # returning the new intent because the user intent changed
            
            response = self.music_info.get_artist_info(artist_name)
            self.store(intent, response)  # Update responses for the music_artist intent


        elif intent == "music_songs":
            song_name = train_songs(self.nlp, statement.text)

            if song_name == None:
                song_name = input("[bot]: I don't have records of that song, but I can search more deeper. Can you type the song name again?\n\n>> ")

                new_intent, similarity, _ = self.get_intent( self.nlp(song_name) )
                self.learning_mode = True

                if similarity < self.min_similarity:   
                    learn_song(song_name)
                
                else:
                    return self.update(  self.nlp(song_name), new_intent )  # returning the new intent because the user intent changed

            response = self.music_info.get_track_info(song_name)
            self.store(intent, response)  # Update responses for the music_songs intent


        elif intent == "music_artist&song":
            artist_name = train_artists(self.nlp, statement.text)
            
            if artist_name == None:
                artist_name = input("[bot]: I don't have records of that artist, but I can search more deeper. Can you type the artist name again?\n\n>> ")
                
                new_intent, similarity, _ = self.get_intent( self.nlp(artist_name) )
                self.learning_mode = True

                if similarity < self.min_similarity:   
                    learn_artist(artist_name)
                
                else:
                    return self.update(  self.nlp(song_name), new_intent )  # returning the new intent because the user intent changed
                

            song_name = train_songs(self.nlp, statement.text)
            if song_name == None:
                song_name = input("[bot]: I don't have records of that song, but I can search more deeper. Can you type the song name again?\n\n>> ")

                new_intent, similarity, _ = self.get_intent( self.nlp(song_name) )
                self.learning_mode = True

                if similarity < self.min_similarity:   
                    learn_song(song_name)
                
                else:
                    return self.update(  self.nlp(song_name), new_intent )  # returning the new intent because the user intent changed
                           

            response = self.music_info.get_artist_song(song_name, artist_name)
            self.store(intent, response)  # Update responses for the music_artist&song intent


        elif intent == "music_tracks_by_artist":
            artist_name = train_artists(self.nlp, statement.text)

            if artist_name == None: 
                artist_name = input("[bot]: I don't have records of that artist, but I can search more deeper. Can you type the artist name again?\n\n>> ")
                
                new_intent, similarity, _ = self.get_intent( self.nlp(artist_name) )
                self.learning_mode = True

                if similarity < self.min_similarity:   
                    learn_artist(artist_name)
                
                else:
                    return self.update(  self.nlp(artist_name), new_intent )  # returning the new intent because the user intent changed

            response = self.music_info.get_tracks_by_artist(artist_name)
            self.store(intent, response)  # Update responses for the music_tracks_by_artist intent

        elif intent == "play_music":

            if self.similarity == 1.0:
                self.spotify.resume()
                return intent
            
            song_by_artist = None
            artist_name = None
            song_name = train_songs(self.nlp, statement.text)

            if song_name == None:
                artist_name = train_artists(self.nlp, statement.text)
                
                if artist_name == None:
                    song_artist = self.spotify.find(track_name=None, artist= statement.text)
                    if song_artist:
                        song_by_artist = song_artist[0]
                        artist_name = song_artist[1]
                        
                        response = self.spotify.play(song_artist[0])
                        self.store(intent, response)  # Update responses for the play_music intent
                        learn_artist(artist_name)

                    else:
                        song_name = input("[bot]: I don't have records of that song, but I can search more deeper. Can you type the song name for me?\n\n>> ")

                        new_intent, similarity, _ = self.get_intent( self.nlp(song_name) )
                        self.learning_mode = True

                        if similarity < self.min_similarity:   
                            learn_song(song_name)
                        
                        else:
                            return self.update(  self.nlp(song_name), new_intent )  # returning the new intent because the user intent changed
                        
                        song_id = self.spotify.find(track_name= song_name)
                        if song_id is None:
                            unknown_spotify_songs = input("[bot]: Can you type the artist name for me?\n\n>> ")

                            new_intent, similarity, _ = self.get_intent( self.nlp(unknown_spotify_songs) )
                            self.learning_mode = True

                            if similarity > self.min_similarity:   
                                return self.update(  self.nlp(unknown_spotify_songs), new_intent )  # returning the new intent because the user intent changed
                            
                            nt = self.spotify.find(song_name, unknown_spotify_songs)

                            if nt is None:
                                response = "I couldn't find your song."
                                self.store(intent, response)  # Update responses for the play_music intent

                            else:
                                track = nt
                                response = self.spotify.play(track)
                                self.store(intent, response)  # Update responses for the play_music intent

                        elif song_id == "exit":
                            intent = "cancel_teaching"
                            return intent 
                        
                        else:
                            response = self.spotify.play(song_id)
                            self.store(intent, response)  # Update responses for the play_music intent

                else:
                    #should work for a artist in the list( yeat)
                    
                    song_artist = self.spotify.find(track_name = None, artist = artist_name)

                    if song_artist is not None:
                        if song_artist == "exit":
                            intent = "cancel_teaching"
                            return intent 
                        
                        response = self.spotify.play(song_artist[0])
                        self.store(intent, response)  # Update responses for the play_music intent
                    else:
                        #caso do trippie redd basicamente em que a msuica mais conhecida é um feat com outro artista
                        response = "I couldn't find your song"
                        self.store(intent, response)  # Update responses for the play_music intent

            else:
                track = self.spotify.find(track_name = song_name)
                if track is None:
                    unknown_spotify_songs = input("[bot]: Can you type the artist name for me?\n\n>> ")

                    new_intent, similarity, _ = self.get_intent( self.nlp(unknown_spotify_songs) )
                    self.learning_mode = True

                    if similarity > self.min_similarity:   
                        return self.update(  self.nlp(unknown_spotify_songs), new_intent )  # returning the new intent because the user intent changed
                    
                    nt = self.spotify.find(song_name,unknown_spotify_songs)
                    if nt is None:
                        response = "I couldn't find your song"
                        self.store(intent, response)  # Update responses for the play_music intent
                    else:
                        track = nt
                        response = self.spotify.play(track)
                        self.store(intent, response)  # Update responses for the play_music intent
                
                else:
                    if track == "exit":
                        intent = "cancel_teaching"
                        return intent 
                    
                    response = self.spotify.play(track)
                    self.store(intent, response)  # Update responses for the play_music intent

            
        elif intent == "pause_music":
            self.spotify.pause()
        
        elif intent == "resume_music":
            self.spotify.resume()

        elif intent == "greeting" and self.username: # if we already know the username, we don't need to ask the username again
            intent = "greeting_with_name"

        elif intent == "cancel_teaching" and not self.learning_mode:
            intent = "teaching_confusion"

        return intent
    

    def get_response(self, intent):
        # Fetch the intent data from the MongoDB document
        intent_data = self.chatbot_db.find_one({}, {intent: 1, "_id": 0})
        
        # Get the responses array for the specific intent
        responses = intent_data[intent]["responses"]
        
        # Choose a random response
        response = random.choice(responses)

        if intent == "greeting_with_name":
            response += f"{self.username}"

        return response
    

    def query(self, statement):
        """ Function that query in the database """
        best_similarity = 0
        index = None

        cursor = self.chatbot_db.find()
        document = next(cursor)["learn"]


        statement = self.nlp( (self.remove_stop_words(statement.text)) )
        for i, knowledge in enumerate(document):
            for pattern in knowledge["patterns"]:

                pattern = self.nlp( (self.remove_stop_words(pattern)) )
                similarity = pattern.similarity(statement)

                if best_similarity < similarity:
                    best_similarity = similarity
                    index = i         

        return best_similarity, index
    
    
    def get_learned_response(self, index):
        cursor = self.chatbot_db.find()
        document = next(cursor)["learn"]
        return random.choice( document[index]["responses"] )
  

    def remove_stop_words(self, pattern):
        pattern = ' '.join(token.text for token in self.nlp((pattern)) if not token.is_punct)

        pattern_lst = pattern.split()
        tokens = [ token for token in pattern_lst if token not in self.stop_words ]
        return ' '.join(tokens)


    def store(self, intent, response, pattern=None):
        
        #print(f"intent = {intent}")
        if intent == "learn":
            new_knowledge = { "patterns":  [pattern], "responses": [response] }
            self.chatbot_db.update_one( {}, {"$push": {intent: new_knowledge}} )

        else:
            # Find the specific intent in the all_intents field and update the responses array for that intent
            update_query = { f"{intent}.responses": [response] }
            self.chatbot_db.update_one({}, {"$set": update_query})


    def exit(self):
        # delete all documents in the collection
        result = self.chatbot_db.delete_many({})
        print("- exit chat bot successfully")



if __name__ == "__main__":
    bot = ConversationalBot()
    bot.run()
    bot.exit()