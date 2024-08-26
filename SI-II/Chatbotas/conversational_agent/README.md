## First Mini-project SI-2: a Conversational Agent

### Introduction and Goals

Conversational agents, sometimes known as chat bots, have several applications, namely for remote automatic assistance. On the other hand, if a conversational agent has a conversational behavior that makes it indistinguishable from the behavior expected in a human being, it may be considered that this agent has human-level intelligence, as Alan Turing suggested. 

The proposed work consists in the development of a conversational agent with the following characteristics:

* Natural language processing (Portuguese and/or English) for some common sentences types.

* Ability to accumulate information/knowledge provided by interlocutors (i.e. learn from interaction) and produce answers to questions.

* For grammatically incorrect sentences, or sentences not supported by the system, react in a ”seemingly intelligent” way.


### How to run the Chat Bot


```bash
# create
python3 -m venv venv

# activate virtual environment
source venv/bin/activate

# install requirements
pip3 install -r requirements.txt

# install language package (for spacy)
python -m spacy download en_core_web_md     # english medium package 

# initialize MongoDB
sudo systemctl start mongod

# run program
python3 main.py
```


### References and Livraries Explored

* https://github.com/gunthercox/ChatterBot/blob/master/README.md 
* https://spacy.io
* https://www.freecodecamp.org/news/creating-a-chat-bot-42861e6a2acd
* https://blog.csml.dev/use-the-actual-eliza-algorithm-in-your-chatbot/