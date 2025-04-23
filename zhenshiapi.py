import requests
import json

from dataclasses import dataclass

@dataclass
class Account:
    username: str
    password: str
    nickname: str
    userPic: str
    androidId: str
    ip: str
    token: str

    @classmethod
    def from_dict(cls, data):
        return cls(
            username=data.get('username', None),
            password=data.get('password', None),
            nickname=data.get('nickname', None),
            userPic=data.get('userPic', None),
            androidId=data.get('androidId', None),
            ip=data.get('ip', None),
            token=data.get('token', None)
        )

default_headers = {
    "Host": "www.xionger.icu:8080",
    "Connection": "Keep-Alive",
    "Accept-Encoding": "gzip",
    "User-Agent": "okhttp/4.9.3",
    "Content-Type": "application/x-www-form-urlencoded"
}

class Zhenshi:
    def __init__(self, account: Account, proxy=None):
        self.account = account
        self.default_headers = default_headers.copy()
        self.proxy = proxy
        self.fakeip = account.ip
        self.Host = f"http://{default_headers['Host']}"
        if account.token is None:
            raise ValueError("Account must be provided")
        self.token = account.token
        self.default_headers["Authorization"] = account.token

    
    def send_request(self, endpoint, headers, data):
        submit_headers = headers.copy()  # Create a copy to avoid shared state issues
        submit_headers["Content-Length"] = str(len(data))
        if self.fakeip:
            submit_headers["X-Forwarded-For"] = self.fakeip
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
        
    def login(self):
        endpoint = "user/login"
        data = f"username={self.account.username}&password={self.account.password}&picurl={self.account.userPic}&nickname={self.account.nickname}&deviceid={self.account.androidId}&ip={self.fakeip}"
        response = self.send_request(endpoint, self.default_headers, data)
        if response and 'data' in response:
            self.token = response['message']
            self.default_headers["Authorization"] = self.token
            return response
        else:
            raise ValueError("Login failed or invalid response")
        
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