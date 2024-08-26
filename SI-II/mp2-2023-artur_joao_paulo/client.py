import sys
import json
import asyncio
import websockets
import getpass
import os

from mapa import Map

class Client:

    # Returns the static game data
    # Obtained from source code
    @staticmethod
    async def recv_game_properties(websocket,agent_name):
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        msg = await websocket.recv()
        game_properties = json.loads(msg)
        return msg, game_properties

    # Returns a representation of the map
    # Obtained from source code
    @staticmethod
    def create_map_representation(game_properties):
        return Map(size=game_properties["size"], mapa=game_properties["map"])

    # Returns the current game state
    # Obtained from source code
    @staticmethod
    async def recv_game_state(websocket):
        try:
            return json.loads(await websocket.recv())
        except websockets.exceptions.ConnectionClosedOK:
            print("Server has cleanly disconnected us")
            return