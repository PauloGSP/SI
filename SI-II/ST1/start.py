from zincbase import KB

kb = KB()

kb.store('is(spike, dog)')
kb.store('is(tom, cat)')
kb.store('is(jerry, rat)')

kb.store('preys_on(spike, cat)')
kb.store('preys_on(tom, rat)')

## [example]
#  is(jerry, rat)             =>     Y, Z 
#  preys_on(tom, rat)         =>     X, Z 
#  fights_against(tom, jerry)  =>     X, Y 

kb.store('fights_against(X,Y) :- is(Y,Z), preys_on(X,Z)')

print(kb.to_triples())      # [('spike', 'is', 'dog'), ('tom', 'is', 'cat'), ('jerry', 'is', 'rat'), ('spike', 'preys_on', 'cat'), ('tom', 'preys_on', 'rat')]

kb.solidify('fights_against')    

print(kb.to_triples())      # [('spike', 'is', 'dog'), ('tom', 'is', 'cat'), ('jerry', 'is', 'rat'), ('spike', 'preys_on', 'cat'), ('tom', 'preys_on', 'rat'), 
                            #  ('spike', 'fights_against', 'tom'), ('tom', 'fights_against', 'jerry')]


# Negative Example
kb.store('~preys_on(jerry, tomatoes)')

# Store from a triple
kb.from_triples([('jerry', 'preys_on', 'cheese')])

# Queries
query = kb.query("preys_on(jerry, Food)")
print(list(query))          # [{'Food': 'cheese'}]

query = kb.query("~preys_on(jerry, Food)")
print(list(query))          # ?


# Neighbors of a node
neighbors_tom = kb.neighbors('tom')
print(neighbors_tom)        # [('cat', [{'pred': 'is'}]), ('rat', [{'pred': 'preys_on'}]), ('jerry', [{'pred': 'fights_against'}])]


# Find a path from start_node to target_node
path = kb.bfs('spike', 'jerry')
print(next(path))


## Métodos que não funcionam
#
# kb.attr('tom', {'isCat': True})  
# kb.plot() 
