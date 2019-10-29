import docker
import os
import time


if __name__ == '__main__':
    game_ids = ['4a4ebc66-f9ce-11e9-b5c3-4cedfb6c1e84', '4a107afe-f9ce-11e9-a769-4cedfb6c1e84']
    ips = ['192.168.86.53', '192.168.86.54']
    ports = [13339, 13340]
    machines = ['test', 'test2']
    client = docker.from_env()

    for machine, port, ip, game_id in zip(machines, ports, ips, game_ids):
        local_volume = f'../rollout_results'
        local_volume = os.path.dirname(os.path.abspath(local_volume))
        local_volume = os.path.join(local_volume, 'rollout_results')
        local_volume = os.path.join(local_volume, game_id)
        container = client.containers.run('dotaservice',
                                          volumes={local_volume: {'bind': '/tmp', 'mode': 'rw'}},
                                          ports={f'{port}/tcp': port},
                                          detach=True)