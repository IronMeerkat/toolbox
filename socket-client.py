import socketio
import base64
import requests

# auth = base64.b64encode('Harry:123456'.encode()).decode()
# bearer = base64.b64encode('Hermione:asdfg'.encode()).decode()
headers = {
    'content-type': 'application/json',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'origin': 'https://hub-dev2.intell-act.com',
    'referer': 'https://hub-dev2.intell-act.com/sign-in',
    'authority': 'hub-dev2.intell-act.com'
}
resp = requests.post('https://hub-dev2.intell-act.com/api/login', json={'_id': 'admin@sky.com', 'password':'3fea869d1cbe341f'}, headers=headers)
# headers.update(resp.headers)
headers['AUTHORIZATION'] = resp.json()['token']

print(resp.json()['token'])

sio = socketio.Client(logger=True)


@sio.on('populate')
def on_populate(data):
    print(data)

@sio.on('update')
def on_update(data):
    print(data)
    

# sio.connect('https://hub-dev2.intell-act.com/timeline', headers=headers)

# sio.emit('request', {'date': '2022-09-05', 'airport': 'AIR'})

sio.connect('http://hub-dev2.intell-act.com/turnaround', headers=headers, namespaces='/turnaround')

sio.emit('request', {'turnaround_id': "2fab3977-8606-4da9-bbda-8affd2c1eabe"}, namespace='/turnaround')
