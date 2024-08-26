import spacy
from spacy.training import Example
from spacy.matcher import PhraseMatcher


def train_artists(nlp, statement):
    matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp.make_doc(artist.lower()) for artist in artists]
    matcher.add("ARTIST", None, *patterns)

    doc = nlp(statement)
    matches = matcher(doc)

    if len(matches) == 0:
        return None

    for match_id, start, end in matches:
        span = doc[start:end]

    return str(span.text)


def learn_artist(artist):
    artists.append(artist)


def train_songs(nlp, statement):

    matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp.make_doc(song.lower()) for song in song_titles]
    matcher.add("SONG", None, *patterns)

    doc = nlp(statement)
    matches = matcher(doc)
    
    if len(matches) == 0:
        """ noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            for song_title in song_titles:
                if nlp(chunk.text.lower()).similarity(nlp(song_title.lower())) >= 0.7:
                    return chunk.text """
        return None

    for match_id, start, end in matches:
        span = doc[start:end]

    return str(span.text)


def learn_song(song):
    if song not in song_titles:
        song_titles.append(song)






artists = ['Playboi Carti', 'Yeat', 'Edith Piaf', 'Charles Aznavour', 'Jacques Brel', 'Johnny Hallyday', 'Serge Gainsbourg', 
    'Yves Montand', 'Maurice Chevalier', 'Édith Piaf', 'Gilbert Bécaud', 'Georges Brassens', 'Nina Simone',
    'Dalida', 'Joe Dassin', 'Juliette Gréco', 'Henri Salvador', 'Mireille Mathieu', 'Céline Dion', 'Ginette Reno',
    'Robert Charlebois', 'Jacques Michel', 'Daniel Lavoie', 'Jean-Pierre Ferland', 'Isabelle Boulay', 'Garou', 
    'Roch Voisine', 'Cirque du Soleil', 'Carla Bruni', 'Zaz', 'Christine and the Queens', 'Indila', 'Stromae', 
    'MC Solaar', 'David Guetta', 'Bob Sinclar', 'Daft Punk', 'Air', 'Gesaffelstein', 'Kavinsky', 'Martin Solveig', 
    'Calvin Harris', 'Deadmau5', 'Tiesto', 'Armin van Buuren', 'Marshmello', 'The Chainsmokers', 'Alan Walker', 
    'Zedd', 'Kygo', 'Skrillex', 'Diplo', 'Beyoncé', 'Taylor Swift', 'Adele', 'Rihanna', 'Justin Bieber',
    'Ed Sheeran', 'Bruno Mars', 'Lady Gaga', 'Katy Perry', 'Ariana Grande', 'Billie Eilish', 'Shawn Mendes', 'Dua Lipa',
    'Post Malone', 'The Weeknd', 'Drake', 'Kendrick Lamar', 'Travis Scott', 'Cardi B', 'Nicki Minaj', 'Jay-Z', 'Kanye West',
    'Lil Nas X', 'Megan Thee Stallion', 'Harry Styles', 'Lizzo', 'Doja Cat', 'Halsey', 'Selena Gomez', 'Camila Cabello', 'Demi Lovato', 'SZA',
    'J. Cole', 'Khalid', 'Imagine Dragons', 'Coldplay', 'Maroon 5', 'The Chainsmokers', 'OneRepublic', 'Panic! At The Disco',
    'Twenty One Pilots', 'Fall Out Boy', 'Green Day', 'Linkin Park', 'My Chemical Romance', 'Nirvana', 'Foo Fighters', 'Metallica',
    'Guns N\' Roses', 'Queen', 'The Beatles', 'The Rolling Stones', 'Pink Floyd', 'Led Zeppelin', 'David Bowie', 'Michael Jackson', 'Madonna', 
    'Prince', 'Whitney Houston', 'Elton John', 'Stevie Wonder', 'Bob Marley', 'James Brown', 'Aretha Franklin', 'Ray Charles',
    'Marvin Gaye', 'Tina Turner', 'Donna Summer', 'Earth Wind & Fire', 'Bee Gees', 'ABBA', 'Gloria Gaynor', 'Boney M',
    'The Jackson 5', 'Chic', 'Kool & The Gang', 'Village People', 'J Balvin', 'Daddy Yankee', 'Ozuna', 'Bad Bunny', 'Maluma', 'Shakira', 'Jennifer Lopez',
    'Enrique Iglesias', 'Luis Fonsi', 'Marc Anthony', 'Carlos Vives', 'BTS', 'BLACKPINK', 'EXO', 'TWICE','Amália Rodrigues', 'Carlos Paredes', 'Dulce Pontes', 
    'António Zambujo', 'Camané', 'Mafalda Veiga', 'Mariza', 'Salvador Sobral', 'Ana Moura', 'Rui Veloso', 'Sérgio Godinho',
    'Madredeus', 'João Pedro Pais', 'Deolinda', 'Gisela João', 'Pedro Abrunhosa', 'Raquel Tavares', 'Linda Martini', 'Xutos e Pontapés',
    'Clã', 'Ornatos Violeta', 'Dead Combo', 'Moonspell', 'David Fonseca', 'The Gift', 'Capitão Fausto', 'Da Weasel',
    'Samuel Úria', 'The Legendary Tigerman', 'Noiserv', 'Silence 4', 'Trovante', 'GNR', 'Heróis do Mar', 'Quinta do Bill', 'Delfins', 'Três Tristes Tigres',
    'A Naifa', 'Banda do Casaco', 'José Afonso', 'Fausto', 'Sérgio Mendes', 'Rão Kyao', 'Vitorino', 'UHF', 'Linda de Suza', 'Tony Carreira', 'Marante',
    'Ágata', 'Santamaria', 'Quim Barreiros', 'Rosinha', 'Leandro', 'Marco Paulo', 'Ana Malhoa', 'Mónica Sintra', 'Beto', 'Anselmo Ralph', 'Guti',
    'Nelson Freitas', 'Fernando Tordo', 'Paulo de Carvalho', 'Simone de Oliveira', 'Lúcia Moniz', 'Filipa Azevedo', 'Rui Bandeira', 'Pedro Ribeiro',
    'Sara Tavares', 'Mísia', 'Camane & Mário Laginha', 'Sétima Legião', 'Sitiados', 'Ritual Tejo', 'Black Company', 'Expensive Soul', 'Buraka Som Sistema',
    'Blasted Mechanism', 'Wraygunn', 'Cool Hipnoise', 'Mão Morta', 'Peste e Sida', 'Ramp', 'Tara Perdida', 'Dama Bete', 'Supernada', 'Linda Martini',
    'Os Pontos Negros', 'Paus', 'B Fachada', 'Peixe:Avião', 'Best Youth', 'Slow J', 'Dino d\'Santiago', 'ProfJam', 'Mundo Segundo', 'Regula', 'NBC', 'Valete','Lon3r Johny'
]

song_titles = [
    "Bohemian Rhapsody", "Stairway to Heaven", "Hotel California", "Imagine", "Smells Like Teen Spirit", "Sweet Child O' Mine", "Billie Jean", 
    "Like a Rolling Stone", "Hey Jude", "Yesterday", "Let It Be", "Purple Haze", "Wonderwall", "My Heart Will Go On", "I Will Always Love You", 
    "Thriller", "Dancing Queen", "Wonderful Tonight", "Landslide", "Tears in Heaven", "Hallelujah", "Every Breath You Take", "Don't Stop Believin'", 
    "Livin' on a Prayer", "Sweet Caroline", "Take on Me", "Everybody Wants to Rule the World", "The Way You Make Me Feel", "Crazy", "Girls Just Want to Have Fun", 
    "Man in the Mirror", "I Want to Hold Your Hand", "I Heard It Through the Grapevine", "My Girl", "Respect", "I Will Survive", "We Will Rock You", 
    "We Are the Champions", "Another One Bites the Dust", "Don't Stop 'Til You Get Enough", "Beat It", "Smooth", "Macarena", "Uptown Funk", "Happy", "I Gotta Feeling",
    "Shape of You", "Despacito", "Hello", "Someone Like You", "Rolling in the Deep", "Single Ladies (Put a Ring on It)", "All of Me", "Thinking Out Loud", 
    "Let Her Go", "Stay with Me", "Shallow", "Bad Guy", "Old Town Road", "Blinding Lights", "Watermelon Sugar", "Save Your Tears", "Levitating", "drivers license", 
    "positions", "Dynamite", "Good 4 U", "Leave the Door Open", "Kiss Me More", "Montero (Call Me By Your Name)", "Peaches", "Deja Vu", "Rockstar", "Savage Love", 
    "Say So", "Don't Start Now", "The Bones", "Memories", "Dance Monkey", "Senorita", "Truth Hurts", "Sunflower", "Without Me", "I Don't Care", "Señorita", "Yummy", 
    "Shape of You", "Perfect", "Sorry", "Love Yourself", "Cold Water", "Starboy", "Rockabye", "That's What I Like", "Wild Thoughts", "Havana", "One Kiss", "In My Feelings", 
    "God's Plan", "Nice for What", "Sad!", "Moonlight", "Changes", "Amar pelos Dois", "A Namorada que Sonhei", "Aquarela do Brasil", "Apenas Mais uma de Amor", 
    "Aquarela", "Atrás da Porta", "Ainda Bem", "Adoro Amar Você", "A Flor e o Espinho", "Além do Horizonte", "Amor Perfeito", "Amor I Love You", "Amiga da Minha Mulher", 
    "Anunciação", "A Estrada", "Apenas um Rapaz Latino Americano", "As Rosas Não Falam", "A Lista", "A Lenda", "Até Quando Esperar", "A Palo Seco", "A Ilha", 
    "Asa Branca", "Alô Alô Marciano", "A Vida do Viajante", "A Saudade é uma Pedra", "A Montanha", "Alma Gêmea", "Alô, Você", "A Paz", "Ainda Lembro", "A Minha Menina", 
    "Anjo", "A Galinha e o Galo", "Amanhã é 23", "Alagados", "Azul da Cor do Mar", "Águas de Março", "Ave Maria", "A Rosa", "Acabou Chorare", "A Menina Dança",
    "A Voz do Morro", "Aquarela Brasileira", "Amor com Amor se Paga", "A Vida Tem Dessas Coisas", "A Distância", "Amor de Índio", "Adeus Batucada", "A Sua",
    "Amor sem Fim", "Alguém me Avisou", "A Casa é Sua", "Aloha", "Asa Morena", "A Paz do Meu Amor", "A Fórmula do Amor", "Amanhã", "Amor I Love You", "A Outra",
    "Amante Profissional", "Aquele Abraço", "Ao Meu Redor", "Apesar de Você", "Ainda Bem que Eu Segui", "Amor Distante", "Agora Só Falta Você", "Anos Dourados",
    "Amado", "A Chave do Perdão", "Aeromoça", "A Luz de Tieta", "A Flor e o Beija-Flor", "A Coisa Mais Linda que Existe", "A Lua e Eu", "Amei Te Ver", "A Volta",
    "A Cura", "A Carta", "A Felicidade", "Aos Nossos Filhos", "Aquarela do Brasil", "Até o Fim", "Amor, Amor", "A Lei do Amor", "Aquele Beijo", "Amor Maior",
    "Aventura", "Amei Demais", "Amar Não é Pecado", "A Ilha", "A Terra Tremeu","Sucesso","Vampire Bites", "Pedras no meu sapato"]