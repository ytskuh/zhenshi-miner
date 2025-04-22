import requests
import json
import hashlib

default_headers = {
    "Host": "www.xionger.icu:8080",
    "Connection": "Keep-Alive",
    "Accept-Encoding": "gzip",
    "User-Agent": "okhttp/4.9.3",
    "Content-Type": "application/x-www-form-urlencoded"
}

def username_to_password(username):
        # 固定的盐值
        tail = "SomeThingAddToUsername7527"
        
        # 拼接用户名和盐值
        salted_username = username + tail
        
        # 使用SHA-256哈希算法
        sha256 = hashlib.sha256()
        sha256.update(salted_username.encode('utf-8'))
        
        # 将哈希值转换为十六进制字符串
        hashed_password = sha256.hexdigest()
        
        return hashed_password

def login(username, picurl, nickname, deviceid, ip):
    url = f"http://{default_headers['Host']}/user/login"
    password = username_to_password(username)
    payload = f"username={username}&password={password}&picurl={picurl}&nickname={nickname}&deviceid={deviceid}&ip={ip}"
    headers = default_headers.copy()
    headers["Content-Length"] = str(len(payload))
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

class Zhenshi:
    def __init__(self, auth = None, proxy = None):
        if auth is None:
            raise ValueError("Account must be provided")
        self.Host = f"http://{default_headers['Host']}"
        self.default_headers = default_headers.copy()
        self.default_headers["Authorization"] = auth
        self.proxy = proxy
    
    def send_request(self, endpoint, headers, data):
        submit_headers = headers.copy()  # Create a copy to avoid shared state issues
        submit_headers["Content-Length"] = str(len(data))
        try:
            response = requests.post(f"{self.Host}/{endpoint}", headers=submit_headers, data=data, proxies=self.proxy)
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        
    def check(self):
        endpoint = "user/checked"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response
        
    def get_userinfo(self):
        endpoint = "user/userInfo"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response

    
    def get_kline(self, code):
        endpoint = "stock/getKLineByCode"
        data = f"code={code}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_orderbook(self, code):
        endpoint = "trade/priceList"
        data = f"code={code}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_userstock(self, code):
        endpoint = "stock/getUserStockByCode"
        data = f"code={code}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_allorder(self):
        endpoint = "trade/getAllOrder"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def addorder(self, code, number, price, operation):
        endpoint = "trade/addOrder"
        data = f"code={code}&number={number}&price={price}&operation={operation}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def cancelorder(self, id):
        endpoint = "trade/cancelOrder"
        data = f"id={id}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def add_lotteryorder(self, number, multiple):
        endpoint = "lottery/addLotteryOrder"
        data = f"number={number}&multiple={multiple}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_userlotteryorder(self):
        endpoint = "lottery/getUserLotteryOrder"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def delete_lotteryorder(self, id):
        endpoint = "lottery/deleteLotteryOrder"
        data = f"id={id}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def reward(self):
        endpoint = "user/reward"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def update_nickname(self, nickname):
        endpoint = "user/updateNickname"
        data = f"nickname={nickname}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_rank(self):
        endpoint = "user/getRank"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response

    def cancel_all_orders(self):
        orders = self.get_allorder()['data']
        for o in orders:
            self.cancelorder(o['id'])