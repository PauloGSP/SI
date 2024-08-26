import json

bot_name = "Orochi"

intents = json.dumps(
{
    "greeting" : {
        "patterns" :  ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "Is anyone there?" ],
        "responses": [f"Hello! What is your name?"]
    },

    "greeting_with_name" : {
        "patterns" : ["hi! i am", "hello! i am", "hey! i am", "good morning! i am", "good afternoon! i am", "good evening! i am",
                      "hi! my name is", "hello! my name is", "hey! my name is", "good morning! my name is", "good afternoon! my name is", "good evening! my name is", "my name is", "i am" ],
        "responses": [f"Nice to meet you, ", "Hello, "]  # add username 
    },

    "goodbye" : {
        "patterns" : ["goodbye", "bye", "good night", "leaving now", "see you later", "exit", "quit"],
        "responses": ["Goodbye!", "See you later :)", "Farewell.", "Talk to you later." ]
    },

    "cancel_teaching" : {
        "patterns" : ["i don't want to teach you", "nevermind", "i don't know", "i was hoping you would know", "no"],
        "responses": ["I'm sorry! Next time I won't let you down.", "No problem!"]
    },

    "teaching_confusion" : {
        "patterns" : [],
        "responses": ["What have I done wrong?"]       # meter mais exemplos
    },

    ################# music

    "music_recommend": {
        "patterns" : ["can you recommend some songs for me?", "find me some good music to listen to.", "give me some songs to listen to", "recommend me some songs"],
        "responses": []  # This will be populated by the update method
    },

    "music_artist": {
        "patterns" : ["can you give me information about", "give me information about", "what do you know about", "who is this artist", "who is", "who are", "have you heard of", "give me information on", "give me info on"],
        "responses": []  # This will be populated by the update method
    },
    
    "music_tracks_by_artist": {
        "patterns" : ["give me some songs of", "give me some tracks of ", "show me some songs by ", "show me some tracks by ", "give me songs of", "recommend me songs of", "can you recommend me songs of", "suggest me a song of", "suggest me songs of"],
        "responses": []  # This will be populated by the update method
    },

    "music_songs": {
        "patterns" : ["do you know this song", "have you listened to", "who is the artist of this song", "do you know the song", "give me details about this song"],
        "responses": [] # This will be populated by the update method
    },

    "music_artist&song": {
        "patterns" : ["give me details about this song by", "detail by", "what do you know of by", "have you listened to by"],
        "responses": [] # This will be populated by the update method
    },

    "play_music":{
        "patterns": ["play music","play a song","play a music","play a track","play track","play song","play","music","track","play by","play from", "play one song of", "continue", "play again", "play the song", "play the artist", "play this artist"],
        "responses": ["Sure!"]
    },

    "pause_music":{
        "patterns": ["pause music", "pause a song", "pause a music", "pause a track", "pause track", "pause song", "pause", "stop"],
        "responses": ["Sure!"]
    },

    "resume_music":{
        "patterns": ["resume music", "resume a song", "resume a music", "resume a track", "resume track", "resume song", "resume", "restart", "continue"],
        "responses": ["Sure!"]
    },

    ################# other

    "options":{
                "patterns": [
                    "functionalities"
                    "what are your functinalities"
                    "options"
                    "how you could help me?",
                    "what can you do?",
                    "what help you provide?",
                    "how you can be helpful?",
                    "what support is offered",
                    "what do you do?",
                    "what is your purpose"
                ],
                "responses": [
                    "I am a music purpose chatbot. My capabilities are: \n\t\t1. I can chat with you. Try asking me for jokes! \n\t\t2. Ask me for information about a song or an artist. \n\t\t3. I can suggest you songs. \n\t\t4. I can give you a couple of songs of a specific artist. \n\t\t5. I can play/pause any song in your spotify."
                ]
    },

    "jokes":{
            "patterns": [
                "tell me a joke",
                "joke",
                "make me laugh"
            ],
            "responses": [
                "A perfectionist walked into a bar...apparently, the bar wasn't set high enough",
                "I ate a clock yesterday, it was very time-consuming",
                "Never criticize someone until you've walked a mile in their shoes. That way, when you criticize them, they won't be able to hear you from that far away. Plus, you'll have their shoes.",
                "The world tongue-twister champion just got arrested. I hear they're gonna give him a really tough sentence.",
                "I own the world's worst thesaurus. Not only is it awful, it's awful.",
                "What did the traffic light say to the car? \"Don't look now, I'm changing.\"",
                "What do you call a snowman with a suntan? A puddle.",
                "How does a penguin build a house? Igloos it together",
                "I went to see the doctor about my short-term memory problems – the first thing he did was make me pay in advance",
                "As I get older and I remember all the people I’ve lost along the way, I think to myself, maybe a career as a tour guide wasn’t for me.",
                "So what if I don't know what 'Armageddon' means? It's not the end of the world."
            ],
    },
    
    "s":{
        "patterns":["faz s"],
        "responses":["Bu sta fabrico"]
    },

    "thanks":{
         "patterns": [
                "thanks",
                "thank you",
                "that's helpful",
                "awesome, thanks",
                "thanks for helping me",
                "you are funny!"
            ],
            "responses": [
                "Happy to help!",
                "Any time!",
                "My pleasure"
            ],

    },
    
    "timer":{
        "patterns": ["set a timer", "timer","set timer","set alarm","alarm"],
        "responses": [""]

    },
    
    "learn" : []
})