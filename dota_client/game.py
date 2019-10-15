import asyncio


class Game:
    ENV_RETRY_DELAY = 15

    def __init__(self, dota_service):
        self.dota_service = dota_service

    async def play(self, config, game_id):

        # Reset and obtain the initial observation. This dictates who we are controlling,
        # this is done before the player definition, because there might be humand playing
        # that take up bot positions.
        config.game_id = game_id
        print(config)
        response = await asyncio.wait_for(self.dota_service.reset(config), timeout=120)

