import requests

base_url = 'http://192.168.1.20:5500/ui/configs'

def get_configs():
    response = requests.get(url=base_url)
    print(response.text)

def edits_configs(local=None, host=None, port=None,
                username=None, password=None,
                cluster_mode=None, tls_mode=None,
                ssl_cert=None, ssl_key=None, ssl_ca=None):

                candidates = {
                    "local": local,
                    "host": host,
                    "port": port,
                    "usernanme": username,
                    "password": password,
                    "cluster_mode": cluster_mode,
                    "tls_mode": tls_mode,
                    "ssl_cert": ssl_cert,
                    "ssl_key": ssl_key,
                    "ssl_ca": ssl_ca,
                    }

                updates = {k: v for k, v in candidates.items() if v is not None}

                if not updates:
                    print("No hay ningún parámetro para actualizar.")
                    return None

                response = requests.post(url=base_url, json=updates)

                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    print(f"Error HTTP: {e} – {response.text}")
                    return None

    
                return response.json()

if __name__ == "__main__":

    get_configs()

    resp = edits_configs(port=8900)

    print(resp)